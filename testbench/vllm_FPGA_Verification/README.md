## 2. vLLM FPGA Verification
```bash
.
├── conftest.py
├── vllm_fpga_verification.py
├── logs
│   ├── verification_log_2024-09-24_19-35-38.log
│   ├── verification_log_2024-09-24_19-48-06.log
├── pytest.ini
├── test_answer.py
├── test_cases.py
├── test_prompt.py
└── vllm_test_text_generation.py
```
#### conftest.py
<pre>
Specifies saved log name format.
</pre>

#### pytest.ini
<pre>
Specifies live log format.
</pre>

#### test_prompt.py
<pre>
Input prompts to verfiy:
  └── short input
  └── long input
</pre>

#### test_answer.py
<pre>
Expected answers to each input prompts for output token length of 100, 200, 400, 600, 800, 1000, 1200.
</pre>

#### cases_answer.py
<pre>
Generates variables for each test case.
Model name, num of devices, output token length, input prompt, golden answer,...are saved in each test case variable.
</pre>

#### vllm_test_text_generation.py
<pre>
This module generates output according to the given test case.
Code is based on /vllm/examples/lpu_inference.py

This module needs: /vllm
  
  
</pre>

#### fpga_verification.py
<pre>
This module compares expected answer and generated output. Logs will be save in ./logs
  
  How to run verification:
  pytest fpga_verification.py                          # prints full info
  pytest fpga_verification.py --tb=no -v               # prints only the summary info
  pytest fpga_verification.py --tb=short               # prints info
  pytest fpga_verification.py --tb=short -k 'test_1'   # select tests
</pre>
