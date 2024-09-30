## 1. Hugging Face FPGA Verification
```bash
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
```
#### conftest.py
```bash
  Specifies log name format
```

#### pytest.ini
```bash
  Specifies live log format
```

#### test_prompt.py
```bash
  Input prompts to verfiy:
    └── short input
    └── long input
```


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

