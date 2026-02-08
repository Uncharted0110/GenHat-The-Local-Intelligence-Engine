from llama_cpp import Llama

# Initialize the model with optimized CPU parameters
llm = Llama(
    model_path="../../models/LFM-1.2B-INT8.gguf",
    n_ctx=4096,           # -c 4096
    n_threads=8,          # -t 8
    n_threads_batch=8,    # -tb 8
    n_batch=256,          # -b 256
    n_ubatch=64,        # Micro-batching support depends on llama-cpp-python version
    flash_attn=True,      # -fa 1
    use_mmap=True,        # --mmap
    type_k=1,             # KV cache type F16
    type_v=1,             # KV cache type F16
    verbose=False,
)

# Conversation history to maintain context
messages = []

while True:
    user_input = input("\n\nYou: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    
    # Add user message to conversation history
    messages.append({"role": "user", "content": user_input})
    
    # Run inference with streaming chat completion
    print("Assistant: ", end="", flush=True)
    response_text = ""
    
    # Stream the response token by token
    for chunk in llm.create_chat_completion(
        messages=messages,
        max_tokens=256,              # Maximum tokens to generate
        temperature=0.7,             # Controls randomness (0.0-2.0)
        top_p=0.9,                   # Nucleus sampling
        top_k=40,                    # Top-k sampling for diversity
        repeat_penalty=1.1,          # Penalize repetition
        stream=True
    ):
        delta = chunk["choices"][0]["delta"]
        if "content" in delta and delta["content"]:
            text = delta["content"]
            response_text += text
            print(text, end="", flush=True)
    
    print()  # Newline after streaming completes
    
    # Add assistant response to conversation history
    messages.append({"role": "assistant", "content": response_text})
    
    # Keep only recent messages if context exceeds limits
    # Estimate ~4 tokens per word, leave buffer for new generation
    total_tokens = sum(len(msg["content"].split()) * 4 for msg in messages)
    if total_tokens > 3500:  # Keep under 4096 limit with safety margin
        messages = messages[-10:]  # Keep last 5 exchanges (user + assistant pairs)