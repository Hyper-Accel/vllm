python -m vllm.entrypoints.openai.api_server --model facebook/opt-1.3b --device fpga --num_lpu_devices 1 --num_gpu_devices 0 --disable_frontend_multiprocessing
