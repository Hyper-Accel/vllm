
# python -m vllm.entrypoints.api_server --model facebook/opt-1.3b --device fpga --num-gpu-devices 0 --num-lpu-devices 1
vllm serve facebook/opt-1.3b --device fpga --num-gpu-devices 0 --num-lpu-devices 1 --disable-frontend-multiprocessing