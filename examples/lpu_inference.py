from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is"
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, top_k=1, min_tokens=30, max_tokens=30)

# Create an LLM.
#llm = LLM(model="facebook/opt-1.3b", device="fpga", pipeline_parallel_size=2)
llm = LLM(model="meta-llama/Meta-Llama-3-8B", device="fpga", tensor_parallel_size=1)
#llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="fpga", tensor_parallel_size=1)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
