## 3. vLLM API Verification

#### File Tree
<pre>
.
├── logs
│   ├── verification_log_2024-09-24_19-35-38.log
│   ├── verification_log_2024-09-24_19-48-06.log
├── conftest.py
├── pytest.ini
├── verification.py
├── test_cases.py
├── test_prompts.py
├── golden_output.py
└── text_generator.py
</pre>
#### conftest.py
<pre>
Specifies saved log name format.
</pre>

#### pytest.ini
<pre>
Specifies live log format. (not necessary)
</pre>

#### test_prompts.py
<pre>
Input prompts used for testing.
</pre>

#### golden_output.py
<pre>
Expected answers to each test cases.
</pre>

#### test_cases.py
<pre>
Defines test cases.
Model name, num of devices, output token length, input prompt, golden answer,...are saved in each test case variable.
</pre>

#### text_generator.py
<pre>
This module generates output according to the given test case.
Code is based on /vllm/examples/lpu_client.py and /vllm/examples/lpu_inference.py
This file opens a server and closes it according to the test case.
(TO DO) Seed is fixed to 7777
This module needs: /vllm
</pre>

#### verification.py
<pre>
This module compares expected answer and generated output. Logs will be save in ./logs
  
## Required commands:
# pip install -U pytest

## How to run verification:
# pytest verification.py                                # prints full info
# pytest verification.py --tb=no -v                     # prints only the summary info
# pytest verification.py --tb=short                     # prints info
# pytest verification.py --tb=short -k 'test_1'         # select tests
# pytest verification.py --tb=no -v --show-capture=no
</pre>
