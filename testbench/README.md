## 1. Hugging Face FPGA Verification
<pre>
.
├── conftest.py
├── fpga_verification.py
├── logs
│   ├── verification_log_2024-09-24_19-35-38.log
│   ├── verification_log_2024-09-24_19-48-06.log
├── pytest.ini
├── test_answer.py
├── test_cases.py
├── test_prompt.py
└── test_text_generation.py
</pre>
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
  Golden answers to each input prompts for 100, 200, 400, 600, 800, 1000, 1200 output token lengths.
</pre>

#### cases_answer.py
<pre>
  Generates variables for each test case.
  Model name, num of devices, output token length, input prompt, golden answer,... are saved in each test case variable.
</pre>

#### test_text_generation.py
<pre>
  Gen
</pre>

#### fpga_verification.py
<pre>
  Generates v
</pre>





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

## 3. vLLM API FPGA Verification
```bash
.
├── conftest.py
├── api_fpga_verification.py
├── logs
│   ├── verification_log_2024-09-24_19-35-38.log
│   ├── verification_log_2024-09-24_19-48-06.log
├── pytest.ini
├── test_answer.py
├── test_cases.py
├── test_prompt.py
└── api_test_text_generation.py
```

