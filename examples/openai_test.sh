python -m vllm.entrypoints.openai.api_server --model facebook/opt-1.3b --device fpga --tensor-parallel-size 1 --port 8000 --disable_frontend_multiprocessing
