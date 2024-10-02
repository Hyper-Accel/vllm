# Pytest

#### Simple Pytest example:
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


<pre>
In the provided code, the pytest.mark.parametrize decorator is used to generate multiple test cases 
by varying the parameters i, j, and k. These parameters correspond to the indices 
for different combinations of models, number of devices, and output lengths from the test_cases object.
</pre>








