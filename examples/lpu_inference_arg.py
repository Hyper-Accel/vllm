from vllm import LLM, SamplingParams
import argparse

# Get arguments
parser = argparse.ArgumentParser(description='vLLM Inference Test Script')
parser.add_argument("-m", "--model", default="facebook/opt-1.3b", type=str, help="name of the language model")
parser.add_argument("-n", "--ncore", default=1, type=int, help="the number of the LPU")
parser.add_argument("-i", "--i_token", default="Hello, my name is", type=str, help="input prompt")
parser.add_argument("-o", "--o_token", default=32, type=int, help="the number of output")
args = parser.parse_args()

# Sample prompts.
prompts = [args.i_token]

# Create a sampling params object and LLM
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, top_k=1, max_tokens=args.o_token)
llm = LLM(model=args.model, device="fpga", tensor_parallel_size=args.ncore)

# Run and print the outputs.
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
