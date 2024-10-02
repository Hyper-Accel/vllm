# Pytest

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


### fpga_verification.py code
```bash
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
```

#### 1. Parametrized Testing:
<pre>
The @pytest.mark.parametrize decorator creates a range of test cases based on all combinations of i, j, and k.
</pre>

#### 2. Test Function:
<pre>
test_text_generation.main() generates an output based on the given test case(test_cases.test_case[i][j][k]).
The output of this generation is compared with the expected answer in logAssert().
</pre>

#### 3. Assertion Logging:
<pre>
The logAssert() function checks whether the test output matches the expected answer(answer_generation).
If they don't match, it logs an error message with details about the test and raises an assertion error.
If the test passes, it logs a success message.
</pre>








