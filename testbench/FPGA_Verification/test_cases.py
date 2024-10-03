##########################################################################
## Test Cases for FPGA Verification
# Test prompts and golden answers are save in test_prompt.py and test_answer.py
##########################################################################

import test_prompt
import test_answer

model = [ 
        #   'falcon-7b',
        #   'phi2-2.7b',
        #   'phi-3.8b',
        #   'llama-30b',
        #   'llama3-8b',
        #   'facebook',
        #   'llama2-7b',
        #   'starcoderbase-1b',
        #   'solar1.1-10.7b',
        #   'llama-7b',
        #   'tinyllama-chat-1.1b',
        #   'megatron-gpt2-345m',
        #   'llama-13b',
        #   'solar-10.7b',
        #   'opt-6.7b',
          'opt-1.3b']


#num_of_devices = [1,2,3,4,5,6,7,8]  #####not working when using more than 1 FPGAs
num_of_devices = [1]
output_len = [100, 200, 400, 600, 800, 1000, 1200]

test_case = [[[0 for _ in range(len(output_len))] for _ in range(len(num_of_devices))] for _ in range(len(model))]

for i in range(len(model)):
    for j in range(len(num_of_devices)):
        for k in range(len(output_len)):    
            test_case[i][j][k] = {
                'model_name'          :   model[i]                          ,
                'num_devices'         :   num_of_devices[j]                 ,   #1~8
                'output_len'          :   output_len[k]                     ,
                'do_sample'           :   False                             ,
                'top_p'               :   0.7                               ,
                'top_k'               :   1                                 ,
                'temperature'         :   0                                 ,
                'repetition_penalty'  :   1.2                               ,
                'prompt'              :   test_prompt.input_s               ,
                'answer'              :   test_answer.input_s[output_len[k]]
            }

