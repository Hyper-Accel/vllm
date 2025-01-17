
## How to install vllm-orion in LPU server

```bash
# 가상환경 새로 생성
conda activate -n vllm-env python==3.10

# hyperdex python package 설치 (conda로 설치하셔야 합니다.)
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 cpuonly -c pytorch
pip install -i https://pypi.hyperaccel.ai/simple hyperdex-transformers==0.0.1+cpu
pip install -i https://pypi.hyperaccel.ai/simple hyperdex-compiler==0.0.1+cpu
pip install -i https://pypi.hyperaccel.ai/simple vllm==0.6.1+cpu

# hyperdex-vllm repo 다운로드 후 example code 실행
conda install -c conda-forge gcc=12.1.0
pip install setuptools_scm
git clone git@github.com:Hyper-Accel/vllm.git
git checkout feature/lpu-backend
python lpu_inference.py
```


## How to install vllm-orion in LPU-GPU hybrid server

```bash
# 가상환경 새로 생성
conda activate -n vllm-env python==3.10

# hyperdex python package 설치
pip install -i https://pypi.hyperaccel.ai/simple hyperdex-transformers==0.0.1+cu121
pip install -i https://pypi.hyperaccel.ai/simple hyperdex-compiler==0.0.1+cu121
pip install -i https://pypi.hyperaccel.ai/simple vllm==0.6.1+cu121
# hyperdex-vllm repo 다운로드 후 example code 실행

conda install -c conda-forge gcc=12.1.0
pip install setuptools_scm
git clone git@github.com:Hyper-Accel/vllm.git
git checkout feature/lpu-backend
python lpu_inference.py
```
