# Pytest

#### Simple Pytest example:
```bash
import pytest

def add(a, b):
    return a + b

def test_add():
    # Test case 1: Check if 1 + 2 equals 3
    assert add(1, 2) == 3

    # Test case 2: Check if -1 + 1 equals 0
    assert add(-1, 1) == 0

    # Test case 3: Check if 0 + 0 equals 0
    assert add(0, 0) == 0
```


<pre>
In the provided code, the pytest.mark.parametrize decorator is used to generate multiple test cases 
by varying the parameters i, j, and k. These parameters correspond to the indices 
for different combinations of models, number of devices, and output lengths from the test_cases object.
</pre>








