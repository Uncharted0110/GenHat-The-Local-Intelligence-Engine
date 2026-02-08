from llama_cpp import Llama

# Initialize the model with optimized CPU parameters
llm = Llama(
    model_path="../../models/LFM-1.2B-INT8.gguf",
    n_ctx=4096,           # -c 4096
    n_threads=8,          # -t 8
    n_threads_batch=8,    # -tb 8
    n_batch=256,          # -b 256
    # n_ubatch=64,        # Micro-batching support depends on llama-cpp-python version
    flash_attn=True,      # -fa 1
    use_mmap=True,        # --mmap
    # Set KV cache types to F16
    # type_k and type_v are often managed automatically, but can be forced if needed
)

while True:
    user_input = input("\n\nEnter your query (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    
    llm.reset()
    # Run inference
    output = llm(
        user_input,
        max_tokens=256,       # -n 256
        echo=True
    )

    print(output["choices"][0]["text"])