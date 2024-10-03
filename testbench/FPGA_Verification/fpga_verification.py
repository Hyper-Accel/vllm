##########################################################################
## Required commands:
# pip install -U pytest

## How to run verification:
# pytest fpga_verification.py                          # prints full info
# pytest fpga_verification.py --tb=no -v               # prints only the summary info
# pytest fpga_verification.py --tb=short               # prints info
# pytest fpga_verification.py --tb=short -k 'test_1'   # select tests
##########################################################################

import pytest
import test_text_generation 
import test_cases
import logging
import logging.handlers

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




@pytest.mark.parametrize("i,j,k", [(i,j,k) for i in range(len(test_cases.model)) for j in range(len(test_cases.num_of_devices)) for k in range(len(test_cases.output_len))])
def test(i,j,k):
    test_generation = test_text_generation.main(test_cases.test_case[i][j][k])
    answer_generation = test_cases.test_case[i][j][k]['answer']
    logAssert(test_generation, answer_generation, [i,j,k])
