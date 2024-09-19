
#python -m vllm.entrypoints.api_server --model facebook/opt-1.3b --device fpga --tensor-parallel-size 2
python -m vllm.entrypoints.api_server --model facebook/opt-1.3b --device fpga --num-gpu-devices 1 --num-lpu-devices 2
#python -m vllm.entrypoints.api_server --model facebook/opt-1.3b --device fpga --num_gpu_devices 1 --num_lpu_devices 2
