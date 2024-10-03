##########################################################################
## LPU Text Generation for Daily Verification
##########################################################################


from vllm import LLM, SamplingParams

class TextGenerator:

    def __init__(self, test_case):
        self.load_done = False
        # Test case
        self.test_case = test_case

    def start(self):
        sampling_params = SamplingParams(temperature=self.test_case["temperature"], \
                                         top_p=self.test_case["top_p"], \
                                         top_k=self.test_case["top_k"], \
                                         repetition_penalty=self.test_case["repetition_penalty"], \
                                         max_tokens=self.test_case["output_len"])

        llm = LLM(model=self.test_case["model_name"], device="fpga", num_lpu_devices=self.test_case["num_devices"], num_gpu_devices=0)
        outputs = llm.generate([self.test_case["prompt"]], sampling_params)
        return outputs[0].outputs[0].text  



def main(test_case):
  generator = TextGenerator(test_case)
  answer = generator.start()
  return answer




# Create a sampling params object.
#sampling_params = SamplingParams(temperature=0.8, top_p=0.8, top_k=1, repetition_penalty=1.2, max_tokens=60)

# Create an LLM.
#llm = LLM(model="facebook/opt-1.3b", device="fpga", pipeline_parallel_size=2)
#llm = LLM(model="meta-llama/Meta-Llama-3-8B", device="fpga", tensor_parallel_size=1)
#llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="fpga", tensor_parallel_size=1)
#llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="fpga", num_lpu_devices=2, num_gpu_devices=0)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
#outputs = llm.generate([test_prompt.input_s], sampling_params)

# Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
