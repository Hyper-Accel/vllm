##########################################################################
## Required commands:
# pip install -U pytest

## How to run verification:
# pytest api_fpga_verification.py                          # prints full info
# pytest api_fpga_verification.py --tb=no -v               # prints only the summary info
# pytest api_fpga_verification.py --tb=short               # prints info
# pytest api_fpga_verification.py --tb=short -k 'test_1'   # select tests
##########################################################################

import pytest
#import test_text_generation 
import api_test_text_generation 
import test_cases
import logging
import logging.handlers
import time



server_process = None


def logAssert(test_generation, answer_generation, test_param):
    test = (test_generation == answer_generation)
    if (not test):
        logging.error(f"[Model: {test_cases.model[test_param[0]]}], " \
                      + f"[# of devices: {test_cases.num_of_devices[test_param[1]]}], " \
                      + f"[Output length: {test_cases.output_len[test_param[2]]}]: \n" \
                      + f"** Expected **: \n{answer_generation} \n** But got **: \n{test_generation}\n")
        assert test, (f"[Model: {test_cases.model[test_param[0]]}], " \
                      + f"[# of devices: {test_cases.num_of_devices[test_param[1]]}], " \
                      + f"[Output length: {test_cases.output_len[test_param[2]]}]: \n" \
                      + f"** Expected **: \n{answer_generation} \n** But got **: \n{test_generation}\n")
    else:
        logging.info(f"[Model: {test_cases.model[test_param[0]]}], [# of devices: {test_cases.num_of_devices[test_param[1]]}], [Output length: {test_cases.output_len[test_param[2]]}]: \nSuccess!!!\n")



# def test_run_back_gnd():
#     server_process = subprocess.Popen(f'python -m vllm.entrypoints.api_server --model {self.test_case["model_name"]} --device fpga --num-gpu-devices 0 --num-lpu-devices {self.test_case["num_devices"]}', \
#                                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, close_fds=True)
    
#     time.sleep(5)
#     return server_process
    # Waiting for server to open
    

@pytest.mark.parametrize("i,j,k", [(i,j,k) for i in range(len(test_cases.model)) for j in range(len(test_cases.num_of_devices)) for k in range(len(test_cases.output_len))])
def test(i,j,k):
    test_generation = api_test_text_generation.main(test_cases.test_case[i][j][k])
    #time.sleep(10)
    answer_generation = test_cases.test_case[i][j][k]['answer']
    logAssert(test_generation, answer_generation, [i,j,k])

