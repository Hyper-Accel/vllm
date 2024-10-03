##########################################################################
## LPU Text Generation for Daily Verification
##########################################################################

import os
import sys
# Set the PATH to include this repo file
package_path = "/".join(__file__.split("/")[:-3])
sys.path.append(package_path)

# Import hyperdex-python
from hyperdex.transformers import AutoModelForCausalLM, AutoTokenizer

##########################################################################
## Main
##########################################################################

def main(test_case):

  # Hyperdex checkpoint path
  hyperdex_ckpt = os.path.join("/data/hyperaccel/model/",test_case["model_name"])

  # Load model and tokenizer
  model = AutoModelForCausalLM.from_pretrained(hyperdex_ckpt, device_map={"gpu": 0, "lpu": test_case["num_devices"]})
  tokenizer = AutoTokenizer.from_pretrained(hyperdex_ckpt)
  
  # Tokenize input context
  inputs = test_case["prompt"]
  input_ids = tokenizer.encode(inputs)

  # Generate text with given test case
  output_ids = model.generate(
    input_ids,
    max_new_tokens=test_case["output_len"],
    # Sampling
    do_sample=test_case['do_sample'],
    top_p=test_case['top_p'],
    top_k=test_case['top_k'],
    temperature=test_case['temperature'],
    repetition_penalty=test_case['repetition_penalty']
  ) 

  # Save only the generated output (remove input prompt)
  output_ids = output_ids[(len(input_ids))+1:] 

  # De-tokenizer output ids
  outputs = tokenizer.decode(output_ids, skip_special_tokens=True)
  return outputs

