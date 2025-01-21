from vllm import LLM, SamplingParams
#from huggingface_hub._login import login

#login(token="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", add_to_git_credential=True)

# Sample prompts.
prompts = ["Act like an experienced HR Manager. Develop a human resources strategy for retaining top talents in a competitive industry. Industry: (e.g Energy( Workforce: (e,g 550) Style: (e.g Formal) Tone: (e.g Convincing)"]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8,
                                 top_p=0.8,
                                 top_k=1,
                                 repetition_penalty=1.2,
                                 max_tokens=60)

# Create an LLM.
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
          device="fpga",
          num_lpu_devices=1,
          num_gpu_devices=0)
#llm = LLM(model="meta-llama/Meta-Llama-3-8B", device="fpga", num_lpu_devices=2, num_gpu_devices=0)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
