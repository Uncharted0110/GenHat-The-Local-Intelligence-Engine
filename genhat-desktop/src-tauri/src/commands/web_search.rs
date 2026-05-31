//! Web search command — zero-setup DuckDuckGo HTML scraping.
//!
//! Uses DuckDuckGo's no-JS HTML endpoint so no API key is ever required.
//! Optional full-page content is fetched via Jina AI Reader (r.jina.ai),
//! which is also free and requires no credentials.
//!
//! All errors are soft: parse or network failures return empty results so
//! the chat flow is never blocked.

use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, ACCEPT_LANGUAGE, USER_AGENT};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

/// A single web search result.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SearchHit {
    pub title: String,
    pub snippet: String,
    pub url: String,
}

/// The full result returned to the frontend.
#[derive(Debug, Serialize, Deserialize)]
pub struct WebSearchResult {
    pub query: String,
    pub results: Vec<SearchHit>,
    /// Pre-formatted context block ready to be injected into the model prompt.
    pub formatted_context: String,
}

// ── DDG HTML scraper ──────────────────────────────────────────────────────────

/// Build headers that look like a real browser request.
fn browser_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        USER_AGENT,
        HeaderValue::from_static(
            "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
        ),
    );
    headers.insert(
        ACCEPT,
        HeaderValue::from_static("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"),
    );
    headers.insert(
        ACCEPT_LANGUAGE,
        HeaderValue::from_static("en-US,en;q=0.5"),
    );
    headers
}

/// Fetch and parse DuckDuckGo HTML search results.
///
/// Returns an empty Vec on any failure — never panics or propagates.
async fn fetch_ddg_results(query: &str, max_results: usize) -> Vec<SearchHit> {
    let client = match reqwest::Client::builder()
        .default_headers(browser_headers())
        .timeout(std::time::Duration::from_secs(10))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            log::warn!("[web_search] Failed to build HTTP client: {e}");
            return vec![];
        }
    };

    let url = format!(
        "https://html.duckduckgo.com/html/?q={}",
        urlencoding::encode(query)
    );

    let html_text = match client.get(&url).send().await {
        Ok(resp) => match resp.text().await {
            Ok(t) => t,
            Err(e) => {
                log::warn!("[web_search] Failed to read DDG response body: {e}");
                return vec![];
            }
        },
        Err(e) => {
            log::warn!("[web_search] DDG request failed: {e}");
            return vec![];
        }
    };

    parse_ddg_html(&html_text, max_results)
}

/// Parse DuckDuckGo HTML into `SearchHit` structs.
fn parse_ddg_html(html: &str, max_results: usize) -> Vec<SearchHit> {
    let document = Html::parse_document(html);

    // DDG HTML result selectors (as of 2024-2026)
    let result_sel = match Selector::parse(".result") {
        Ok(s) => s,
        Err(_) => return vec![],
    };
    let title_sel = match Selector::parse(".result__a") {
        Ok(s) => s,
        Err(_) => return vec![],
    };
    let snippet_sel = match Selector::parse(".result__snippet") {
        Ok(s) => s,
        Err(_) => return vec![],
    };
    let url_sel = match Selector::parse(".result__url") {
        Ok(s) => s,
        Err(_) => return vec![],
    };

    let mut hits: Vec<SearchHit> = Vec::new();

    for result in document.select(&result_sel) {
        if hits.len() >= max_results {
            break;
        }

        let title = result
            .select(&title_sel)
            .next()
            .map(|el| el.text().collect::<String>().trim().to_owned())
            .unwrap_or_default();

        if title.is_empty() {
            continue;
        }

        let snippet = result
            .select(&snippet_sel)
            .next()
            .map(|el| el.text().collect::<String>().trim().to_owned())
            .unwrap_or_default();

        // Try to get a clean URL from the visible .result__url span.
        // DDG wraps the real href in a redirect; the visible text is usually cleaner.
        let url = result
            .select(&url_sel)
            .next()
            .map(|el| {
                let raw = el.text().collect::<String>().trim().to_owned();
                if raw.starts_with("http") {
                    raw
                } else {
                    format!("https://{raw}")
                }
            })
            // fallback: pull href from the title link and strip DDG redirect prefix
            .or_else(|| {
                result.select(&title_sel).next().and_then(|el| {
                    el.value().attr("href").map(|href| {
                        // DDG links look like "/l/?uddg=https%3A%2F%2F..."
                        if let Some(pos) = href.find("uddg=") {
                            let encoded = &href[pos + 5..];
                            urlencoding::decode(encoded)
                                .map(|s| s.into_owned())
                                .unwrap_or_else(|_| href.to_owned())
                        } else if href.starts_with("http") {
                            href.to_owned()
                        } else {
                            format!("https://duckduckgo.com{href}")
                        }
                    })
                })
            })
            .unwrap_or_default();

        hits.push(SearchHit { title, snippet, url });
    }

    hits
}

// ── Jina AI Reader (full-page content) ───────────────────────────────────────

/// Fetch full-page markdown content for a URL via `r.jina.ai`.
///
/// Returns `None` on any failure — caller should fall back to snippet.
async fn fetch_jina_content(url: &str) -> Option<String> {
    let jina_url = format!("https://r.jina.ai/{url}");

    let client = match reqwest::Client::builder()
        .default_headers(browser_headers())
        .timeout(std::time::Duration::from_secs(15))
        .build()
    {
        Ok(c) => c,
        Err(_) => return None,
    };

    match client.get(&jina_url).send().await {
        Ok(resp) => match resp.text().await {
            Ok(text) => {
                // Truncate to ~2000 chars to keep context lean for small models
                let truncated: String = text.chars().take(2000).collect();
                Some(truncated)
            }
            Err(_) => None,
        },
        Err(e) => {
            log::warn!("[web_search] Jina fetch failed for {url}: {e}");
            None
        }
    }
}

// ── Context formatter ─────────────────────────────────────────────────────────

fn format_context(results: &[SearchHit]) -> String {
    if results.is_empty() {
        return String::new();
    }

    let mut ctx = String::from(
        "Use the following up-to-date web search results to inform your answer. \
         Cite sources where relevant.\n\nWeb Search Results:\n",
    );

    for (i, hit) in results.iter().enumerate() {
        ctx.push_str(&format!(
            "\n[{}] {}\nURL: {}\n{}\n",
            i + 1,
            hit.title,
            hit.url,
            hit.snippet
        ));
    }

    ctx
}

// ── Tauri command ─────────────────────────────────────────────────────────────

/// Search the web and return structured results + a formatted context block.
///
/// * `query` — the user's message (trimmed to 150 chars before sending to DDG).
/// * `max_results` — how many results to return (typically 5 for snippets, 2 for full).
/// * `fetch_content` — if true, fetches full-page markdown via Jina AI Reader for each result.
///
/// This command never returns an Err that blocks chat: parse/network failures
/// produce an empty `results` list with an empty `formatted_context`.
#[tauri::command]
pub async fn web_search(
    query: String,
    max_results: u32,
    fetch_content: bool,
) -> Result<WebSearchResult, String> {
    let trimmed_query: String = query.chars().take(150).collect();
    let n = (max_results.min(10)).max(1) as usize;

    let mut results = fetch_ddg_results(&trimmed_query, n).await;

    if fetch_content && !results.is_empty() {
        // Enrich each result with full page content
        let enriched_futures: Vec<_> = results
            .iter()
            .map(|hit| fetch_jina_content(&hit.url))
            .collect();

        let contents = futures_util::future::join_all(enriched_futures).await;

        for (hit, content_opt) in results.iter_mut().zip(contents) {
            if let Some(content) = content_opt {
                // Replace brief snippet with richer page content
                hit.snippet = content;
            }
        }
    }

    let formatted_context = format_context(&results);

    Ok(WebSearchResult {
        query: trimmed_query,
        results,
        formatted_context,
    })
}
