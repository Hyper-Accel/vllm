from vllm import LLM, SamplingParams
from huggingface_hub._login import login


# Sample prompts.
prompts = ["Hello, my name is"]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.8, top_k=1, repetition_penalty=1.2, max_tokens=60)

# Create an LLM.
llm = LLM(model="facebook/opt-1.3b", device="fpga", num_lpu_devices=1, num_gpu_devices=0)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
