# Pytest (What you need to know)

### Simple Pytest example:
```bash
import pytest

def add(a, b):
    return a + b

def test_add():
    # Test case 1: Check if 1 + 2 equals 3
    assert add(1, 2) == 3

    # Test case 2: Check if -1 + 1 equals 2 (This is intentionally incorrect)
    assert add(-1, 1) == 2

    # Test case 3: Check if 0 + 0 equals 0 : PASS
    assert add(0, 0) == 0
```
Output:
<pre>
============================= test session starts ==============================
collected 1 item

pytest_exampple.py F..                                         [100%]

=================================== FAILURES ====================================
__________________________________ test_add _____________________________________

    def test_add():
        # Test case 1: Check if 1 + 2 equals 3
        assert add(1, 2) == 3

        # Test case 2: Check if -1 + 1 equals 2 (This is intentionally incorrect)
>       assert add(-1, 1) == 2  # This will fail because the correct answer is 0
E       assert 0 == 2
E        +  where 0 = add(-1, 1)

test_simple_calculator.py:9: AssertionError
=========================== short test summary info ============================
FAILED test_simple_calculator.py::test_add - assert 0 == 2
============================== 1 failed, 2 passed in 0.02s ======================
</pre>


### verification.py code
```bash
def logAssert(test_generation, answer_generation, test_num, gen_time):

    # Check Time Out
    if(gen_time > 20):
        logging.error(f"\nTime Out!: {gen_time:.5f} seconds.\n")

    test = (test_generation == answer_generation)
    if (not test):
        #Log output when failed
        logging.error(f"\nFAIL: {gen_time:.5f} seconds. \n" \
                      + f"[Model: {test_cases.test_case[test_num]['model_name']}], \n" \
                      + f"[# of LPUs: {test_cases.test_case[test_num]['num_of_LPU']}], \n" \
                      + f"[# of GPUs: {test_cases.test_case[test_num]['num_of_GPU']}], \n" \
                      + f"[Output length: {test_cases.test_case[test_num]['output_len']}], \n" \
                      + f"[Do Sample: {test_cases.test_case[test_num]['do_sample']}], \n" \
                      + f"[Top_p: {test_cases.test_case[test_num]['top_p']}], \n" \
                      + f"[Top_k: {test_cases.test_case[test_num]['top_k']}], \n" \
                      + f"[Temperature: {test_cases.test_case[test_num]['temperature']}], \n" \
                      + f"[Repetition_penalty: {test_cases.test_case[test_num]['repetition_penalty']}] \n" \
                      + f"** Expected **: \n{answer_generation} \n** But got **: \n{test_generation}\n")

        #Live console output when failed (also necessary for summarization logging)
        assert test, (f"\nFAIL: {gen_time:.5f} seconds. \n" \
                      + f"[Model: {test_cases.test_case[test_num]['model_name']}], \n" \
                      + f"[# of LPUs: {test_cases.test_case[test_num]['num_of_LPU']}], \n" \
                      + f"[# of GPUs: {test_cases.test_case[test_num]['num_of_GPU']}], \n" \
                      + f"[Output length: {test_cases.test_case[test_num]['output_len']}], \n" \
                      + f"[Do Sample: {test_cases.test_case[test_num]['do_sample']}], \n" \
                      + f"[Top_p: {test_cases.test_case[test_num]['top_p']}], \n" \
                      + f"[Top_k: {test_cases.test_case[test_num]['top_k']}], \n" \
                      + f"[Temperature: {test_cases.test_case[test_num]['temperature']}], \n" \
                      + f"[Repetition_penalty: {test_cases.test_case[test_num]['repetition_penalty']}] \n" \
                      + f"** Expected **: \n{answer_generation} \n** But got **: \n{test_generation}\n")

    else:
        #Log output when passed
        logging.info(f"\nPASS: {gen_time:.5f} seconds. \n" \
                      + f"[Model: {test_cases.test_case[test_num]['model_name']}], \n" \
                      + f"[# of LPUs: {test_cases.test_case[test_num]['num_of_LPU']}], \n" \
                      + f"[# of GPUs: {test_cases.test_case[test_num]['num_of_GPU']}], \n" \
                      + f"[Output length: {test_cases.test_case[test_num]['output_len']}], \n" \
                      + f"[Do Sample: {test_cases.test_case[test_num]['do_sample']}], \n" \
                      + f"[Top_p: {test_cases.test_case[test_num]['top_p']}], \n" \
                      + f"[Top_k: {test_cases.test_case[test_num]['top_k']}], \n" \
                      + f"[Temperature: {test_cases.test_case[test_num]['temperature']}], \n" \
                      + f"[Repetition_penalty: {test_cases.test_case[test_num]['repetition_penalty']}] \n" \
                      + f"Pass!!!\n")
```

#### 1. Parametrized Testing:
<pre>
The @pytest.mark.parametrize decorator allows Pytesting on more than one test cases.
You may add more test cases in test_cases.py.
</pre>

#### 2. Test Function:
<pre>
text_generator.py generates an output based on the given test case(test_cases.test_case[test_num]).
The output of this generation is compared with the expected answer(golden answer) in logAssert().
</pre>

#### 3. Assertion Logging:
<pre>
The logAssert() function checks whether the test output matches the expected answer(answer_generation).
If they don't match, it logs an error message with details about the test and raises an assertion error.
If the test passes, it logs a success message.
</pre>

#### 4. Test Case Compositions:
<pre>
'''
test_case.insert(0,{
    'model_name'          :   model[0]                          ,
    'num_of_LPU'          :   num_of_LPU[0]                     , 
    'num_of_GPU'          :   num_of_GPU[0]                     , 
    'output_len'          :   output_len[0]                     ,
    'top_p'               :   ttt[0][0]                         ,
    'top_k'               :   ttt[0][1]                         ,
    'temperature'         :   ttt[0][2]                         ,
    'repetition_penalty'  :   repetition_penalty[0]             ,
    'Input Prompt'        :   test_prompts.prompt[0]            ,
    'Golden Output'       :   golden_output.test_case[0]        
})
'''
</pre>










