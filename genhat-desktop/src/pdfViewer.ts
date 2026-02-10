// PDF Viewer with text selection support using PDF.js
// PDF.js is loaded globally from CDN in index.html

// Type declarations for PDF.js global
declare const pdfjsLib: any
declare const lucide: any

export interface PDFViewerOptions {
  container: HTMLElement
  onTextSelected?: (text: string, pageNumber: number) => void
  onError?: (error: Error) => void
}

export class PDFViewer {
  private container: HTMLElement
  private pdf: any = null
  private currentPage: number = 1
  private scale: number = 1.5
  private onTextSelected?: (text: string, pageNumber: number) => void
  private onError?: (error: Error) => void

  constructor(options: PDFViewerOptions) {
    this.container = options.container
    this.onTextSelected = options.onTextSelected
    this.onError = options.onError
  }

  async loadPDF(file: File): Promise<void> {
    try {
      // Clear container
      this.container.innerHTML = '<div style="text-align: center; padding: 20px; color: #888;">Loading PDF...</div>'

      // Read file as ArrayBuffer
      const arrayBuffer = await file.arrayBuffer()
      const typedArray = new Uint8Array(arrayBuffer)

      // Load PDF
      const loadingTask = pdfjsLib.getDocument({ data: typedArray })
      this.pdf = await loadingTask.promise

      // Create viewer UI
      this.createViewerUI()

      // Render all pages
      await this.renderAllPages()
    } catch (error) {
      console.error('Error loading PDF:', error)
      if (this.onError) {
        this.onError(error instanceof Error ? error : new Error('Failed to load PDF'))
      }
      this.container.innerHTML = `
        <div style="text-align: center; padding: 40px; color: #ff6b6b;">
          <div style="font-size: 48px; margin-bottom: 12px;">⚠️</div>
          <p>Failed to load PDF</p>
        </div>
      `
    }
  }

  private createViewerUI(): void {
    if (!this.pdf) return

    this.container.innerHTML = `
      <style>
        #pdfCanvasContainer::-webkit-scrollbar {
          width: 10px;
          background: #1a1a1a;
        }
        #pdfCanvasContainer::-webkit-scrollbar-track {
          background: #1a1a1a;
        }
        #pdfCanvasContainer::-webkit-scrollbar-thumb {
          background: #ff8c00;
          border-radius: 5px;
          border: 2px solid #1a1a1a;
        }
        #pdfCanvasContainer::-webkit-scrollbar-thumb:hover {
          background: #ff6b00;
        }
        .pdf-page-container {
          position: relative;
          background: white;
          box-shadow: 0 4px 12px rgba(0,0,0,0.5);
          margin-bottom: 20px;
        }
        /* Hide spin buttons for number input */
        #pageInput::-webkit-outer-spin-button,
        #pageInput::-webkit-inner-spin-button {
          -webkit-appearance: none;
          margin: 0;
        }
        #pageInput {
          -moz-appearance: textfield;
        }
      </style>
      <div style="display: flex; flex-direction: column; height: 100%; background: #1a1a1a;">
        <!-- Toolbar -->
        <div id="pdfToolbar" style="display: flex; align-items: center; justify-content: space-between; padding: 12px; background: #0d0d0d; border-bottom: 1px solid #2a2a2a;">
          <div style="display: flex; align-items: center; gap: 8px;">
            <span style="color: #e0e0e0; font-size: 14px;">Page</span>
            <input type="number" id="pageInput" value="1" min="1" max="${this.pdf.numPages}" style="background: #2a2a2a; border: 1px solid #3a3a3a; color: #e0e0e0; padding: 4px 8px; border-radius: 4px; width: 50px; text-align: center;">
            <span style="color: #e0e0e0; font-size: 14px;">of ${this.pdf.numPages}</span>
          </div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <button id="zoomOut" style="background: #2a2a2a; border: 1px solid #3a3a3a; color: #e0e0e0; padding: 6px 12px; border-radius: 4px; cursor: pointer; display: flex; align-items: center; justify-content: center;"><i data-lucide="minus" style="width: 16px; height: 16px;"></i></button>
            <span id="zoomLevel" style="color: #e0e0e0; font-size: 14px;">${Math.round(this.scale * 100)}%</span>
            <button id="zoomIn" style="background: #2a2a2a; border: 1px solid #3a3a3a; color: #e0e0e0; padding: 6px 12px; border-radius: 4px; cursor: pointer; display: flex; align-items: center; justify-content: center;"><i data-lucide="plus" style="width: 16px; height: 16px;"></i></button>
          </div>
        </div>
        
        <!-- Canvas container -->
        <div id="pdfCanvasContainer" style="flex: 1; overflow-y: auto; overflow-x: hidden; display: flex; flex-direction: column; align-items: center; padding: 20px; background: #1a1a1a;">
          <div id="pagesWrapper"></div>
        </div>
      </div>
    `

    // Initialize icons
    if (typeof lucide !== 'undefined') {
      lucide.createIcons({
        root: this.container,
        nameAttr: 'data-lucide'
      })
    }

    // Setup event listeners
    this.setupEventListeners()
  }

  private setupEventListeners(): void {
    const zoomInBtn = this.container.querySelector('#zoomIn') as HTMLButtonElement
    const zoomOutBtn = this.container.querySelector('#zoomOut') as HTMLButtonElement
    const pageInput = this.container.querySelector('#pageInput') as HTMLInputElement
    const pdfCanvasContainer = this.container.querySelector('#pdfCanvasContainer') as HTMLDivElement

    if (zoomInBtn) {
      zoomInBtn.addEventListener('click', () => this.zoomIn())
    }

    if (zoomOutBtn) {
      zoomOutBtn.addEventListener('click', () => this.zoomOut())
    }

    if (pageInput) {
      pageInput.addEventListener('change', () => {
        const pageNum = parseInt(pageInput.value)
        if (pageNum >= 1 && pageNum <= this.pdf.numPages) {
          this.scrollToPage(pageNum)
        }
      })
      
      pageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
          const pageNum = parseInt(pageInput.value)
          if (pageNum >= 1 && pageNum <= this.pdf.numPages) {
            this.scrollToPage(pageNum)
            pageInput.blur()
          }
        }
      })
    }

    if (pdfCanvasContainer) {
      pdfCanvasContainer.addEventListener('scroll', () => {
        this.updateCurrentPageFromScroll()
      })
    }
  }

  private async renderAllPages(): Promise<void> {
    if (!this.pdf) return

    const pagesWrapper = this.container.querySelector('#pagesWrapper') as HTMLDivElement
    if (!pagesWrapper) return
    
    pagesWrapper.innerHTML = '' // Clear existing pages

    for (let pageNum = 1; pageNum <= this.pdf.numPages; pageNum++) {
      try {
        const page = await this.pdf.getPage(pageNum)
        const viewport = page.getViewport({ scale: this.scale })

        // Create page container
        const pageContainer = document.createElement('div')
        pageContainer.className = 'pdf-page-container'
        pageContainer.id = `page-container-${pageNum}`
        pageContainer.setAttribute('data-page-number', pageNum.toString())
        pageContainer.style.width = `${viewport.width}px`
        pageContainer.style.height = `${viewport.height}px`
        pageContainer.style.setProperty('--scale-factor', this.scale.toString())
        
        // Create canvas
        const canvas = document.createElement('canvas')
        canvas.height = viewport.height
        canvas.width = viewport.width
        
        // Create text layer
        const textLayer = document.createElement('div')
        textLayer.className = 'textLayer'
        textLayer.style.position = 'absolute'
        textLayer.style.left = '0'
        textLayer.style.top = '0'
        textLayer.style.right = '0'
        textLayer.style.bottom = '0'
        textLayer.style.overflow = 'hidden'
        textLayer.style.opacity = '1'
        textLayer.style.lineHeight = '1.0'
        textLayer.style.width = `${viewport.width}px`
        textLayer.style.height = `${viewport.height}px`

        pageContainer.appendChild(canvas)
        pageContainer.appendChild(textLayer)
        pagesWrapper.appendChild(pageContainer)

        // Render page
        const context = canvas.getContext('2d')
        if (context) {
          const renderContext = {
            canvasContext: context,
            viewport: viewport
          }
          await page.render(renderContext).promise
        }

        // Render text layer
        await this.renderTextLayer(page, viewport, textLayer)
        
        // Add text selection listener for this page
        textLayer.addEventListener('mouseup', () => {
          const selection = window.getSelection()
          if (selection && selection.toString().trim().length > 0) {
            const selectedText = selection.toString().trim()
            if (this.onTextSelected) {
              this.onTextSelected(selectedText, pageNum)
            }
          }
        })

      } catch (error) {
        console.error(`Error rendering page ${pageNum}:`, error)
      }
    }
  }

  private async renderTextLayer(
    page: any,
    viewport: any,
    container: HTMLDivElement
  ): Promise<void> {
    // Clear existing text layer
    container.innerHTML = ''

    try {
      // Use PDF.js built-in text layer renderer for proper metrics and selection
      const textContent = await page.getTextContent({ disableCombineTextItems: false })
      const renderTask = pdfjsLib.renderTextLayer({
        textContentSource: textContent,
        container,
        viewport,
        textDivs: [],
        enhanceTextSelection: true
      })
      if (renderTask && renderTask.promise) {
        await renderTask.promise
      }

      // Make the glyphs transparent but keep native selection highlight fully visible
      // We avoid forcing a font family or manual transforms to preserve accurate positioning
      container.style.opacity = '0.5'
      container.style.color = 'transparent'
      container.style.userSelect = 'text'
    } catch (error) {
      console.error('Error rendering text layer:', error)
    }
  }

  private async zoomIn(): Promise<void> {
    this.scale = Math.min(this.scale + 0.25, 3)
    this.updateZoomLevel()
    await this.renderAllPages()
  }

  private async zoomOut(): Promise<void> {
    this.scale = Math.max(this.scale - 0.25, 0.5)
    this.updateZoomLevel()
    await this.renderAllPages()
  }

  private updateZoomLevel(): void {
    const zoomLevel = this.container.querySelector('#zoomLevel')
    if (zoomLevel) {
      zoomLevel.textContent = `${Math.round(this.scale * 100)}%`
    }
  }

  private scrollToPage(pageNum: number): void {
    const pageContainer = this.container.querySelector(`#page-container-${pageNum}`)
    if (pageContainer) {
      pageContainer.scrollIntoView({ behavior: 'smooth', block: 'start' })
      this.currentPage = pageNum
    }
  }

  private updateCurrentPageFromScroll(): void {
    const pdfCanvasContainer = this.container.querySelector('#pdfCanvasContainer') as HTMLDivElement
    if (!pdfCanvasContainer) return

    const pageContainers = this.container.querySelectorAll('.pdf-page-container')
    let visiblePage = 1
    let maxVisibility = 0

    pageContainers.forEach((container) => {
      const rect = container.getBoundingClientRect()
      const containerRect = pdfCanvasContainer.getBoundingClientRect()
      
      // Calculate intersection
      const intersectionTop = Math.max(rect.top, containerRect.top)
      const intersectionBottom = Math.min(rect.bottom, containerRect.bottom)
      const height = Math.max(0, intersectionBottom - intersectionTop)
      
      if (height > maxVisibility) {
        maxVisibility = height
        const pageNum = parseInt(container.getAttribute('data-page-number') || '1')
        visiblePage = pageNum
      }
    })
    
    if (this.currentPage !== visiblePage) {
      this.currentPage = visiblePage
      const pageInput = this.container.querySelector('#pageInput') as HTMLInputElement
      if (pageInput && document.activeElement !== pageInput) {
        pageInput.value = visiblePage.toString()
      }
    }
  }

  destroy(): void {
    this.container.innerHTML = ''
    this.pdf = null
  }
}
