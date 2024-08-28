from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "1 2 3 4 5",
#    "Do you know KAIST?",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, top_k=1, min_tokens=30, max_tokens=30)

# Create an LLM.
llm = LLM(model="facebook/opt-1.3b", device="fpga")
#llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="fpga")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
print((outputs))
print((outputs[0].prompt))
print((outputs[0].outputs[0]))
print((outputs[0].outputs[0].text))
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
