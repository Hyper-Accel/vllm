## HyperDex-vLLM

HyperDex supports the vLLM framework to run on LPU(LLM Processing Unit). As you know, the vLLM framework officially supports a variety of hardware including GPU, TPU, and XPU. HyperDex has its own branch of vLLM with a backend specifically designed for LPU, making it very easy to use. If your system is already using vLLM, you can switch hardware from GPU to LPU without changing any code. Then, let's jump into the hyperdex-vllm!


Requirements
- vLLM.0.5.5
- libtorch.2.4.0
- hyperdex.1.3.2

Installation
```bash
cd scripts
./install_script.sh
```

Simple Execution using vLLM API
In our branch, you can easily execute LPU by setting the option `device=fpga` and `num_lpu_devices=1`. Try set the option `num_gpu_devices=1` if you want to test hybrid mode.
If you aren't set the option `device(default:cuda)`, vLLM functions like original vLLM.

```bash
cd examples
python lpu_inference.py
```


Execution Serving API
```bash
# Open the serving system
cd examples
./vllm_serve.sh

# Send requests for serving system from another terminal
cd examples
python lpu_client.py
```

Visit our [website](https://docs.hyperaccel.ai) to learn more.

