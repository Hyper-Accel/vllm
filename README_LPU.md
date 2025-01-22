# 1. vLLM-lpu설치 가이드

## LPU 서버에서 vllm 설치 가이드

### 1. Conda를 사용한 설치 방법 (PyPI 인증 필요)

```bash
# 1. 가상환경 생성 및 활성화
conda create -n vllm-env python=3.10
conda activate vllm-env

# 2. PyTorch CPU 버전 설치
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 cpuonly -c pytorch

# 3. Hyperdex 패키지 설치 (PyPI 인증 정보 필요)
pip install -i https://<username>:<password>@pypi.hyperaccel.ai/simple/ hyperdex-transformers==0.0.2+cpu
pip install -i https://<username>:<password>@pypi.hyperaccel.ai/simple/ hyperdex-compiler==0.0.1+cpu

# 4. 빌드 의존성 설치
pip install -r requirements-build.txt

# 5. vllm 설치 (-e 옵션은 현재 디렉토리에 설치하는 옵션, site-packages에 설치하고 싶을 시 -e 옵션 제거)
VLLM_TARGET_DEVICE=fpga pip install -e . --no-build-isolation
```

### 2. Pip를 사용한 설치 방법 (PyPI 인증 필요)

```bash
# 1. 가상환경 생성 및 활성화
python -m venv vllm-env
source vllm-env/bin/activate

# 2. pytorch 설치
pip install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 -i https://download.pytorch.org/whl/cpu

# 3. Hyperdex 패키지 설치 (PyPI 인증 정보 필요)
pip install -i https://<username>:<password>@pypi.hyperaccel.ai/simple/ hyperdex-transformers==0.0.2+cpu
pip install -i https://<username>:<password>@pypi.hyperaccel.ai/simple/ hyperdex-compiler==0.0.1+cpu

# 4. 빌드 의존성 설치
pip install -r requirements-build.txt

# 5. vllm 설치 (-e 옵션은 현재 디렉토리에 설치하는 옵션, site-packages에 설치하고 싶을 시 -e 옵션 제거)
VLLM_TARGET_DEVICE=fpga pip install -e . --no-build-isolation
```

## LPU-GPU 하이브리드 서버에서 설치 방법 (CUDA 12.1 필요)

### 1. Conda를 사용한 설치 방법 (PyPI 인증 필요)

```bash
# 1. 가상환경 생성 및 활성화
conda create -n vllm-env python=3.10
conda activate vllm-env

# 2. pytorch 설치
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. Hyperdex CUDA 패키지 설치 (PyPI 인증 정보 필요)
pip install -i https://<username>:<password>@pypi.hyperaccel.ai/simple/ hyperdex-transformers==0.0.2+cu121
pip install -i https://<username>:<password>@pypi.hyperaccel.ai/simple/ hyperdex-compiler==0.0.1+cu121

# 4. 빌드 의존성 설치
pip install -r requirements-build.txt

# 5. vllm 설치 (-e 옵션은 현재 디렉토리에 설치하는 옵션, site-packages에 설치하고 싶을 시 -e 옵션 제거)
VLLM_TARGET_DEVICE=fpga pip install -e . --no-build-isolation
```

### 2. Pip를 사용한 설치 방법 (PyPI 인증 필요)

```bash
# 1. 가상환경 생성 및 활성화
python -m venv vllm-env
source vllm-env/bin/activate

# 2. pytorch 설치
pip install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 -i https://download.pytorch.org/whl/cu121

# 3. Hyperdex CUDA 패키지 설치 (PyPI 인증 정보 필요)
pip install -i https://<username>:<password>@pypi.hyperaccel.ai/simple/ hyperdex-transformers==0.0.2+cu121
pip install -i https://<username>:<password>@pypi.hyperaccel.ai/simple/ hyperdex-compiler==0.0.1+cu121

# 4. 빌드 의존성 설치
pip install -r requirements-build.txt

# 5. vllm 설치 (-e 옵션은 현재 디렉토리에 설치하는 옵션, site-packages에 설치하고 싶을 시 -e 옵션 제거)
VLLM_TARGET_DEVICE=fpga pip install -e . --no-build-isolation
```

# 2. vLLM-lpu 실행 가이드

## 1. 단일 추론 실행 (LLMEngine)

```bash
# 기본 실행 방법
python examples/lpu_inference_arg.py -m [모델명] -l [LPU 개수] -g [GPU 개수] -i [입력 텍스트] -o [출력 토큰 수]

# 예시
python examples/lpu_inference_arg.py -m facebook/opt-1.3b -l 1 -g 0 -i "Hello, my name is" -o 32

# 파라미터 설명
# -m, --model: 허깅페이스 모델 이름 (예: facebook/opt-1.3b, TinyLlama/TinyLlama-1.1B-Chat-v1.0)
# -l, --num-lpu-devices: 사용할 LPU 디바이스 수
# -g, --num-gpu-devices: 사용할 GPU 디바이스 수
# -i, --input: 입력 프롬프트
# -o, --max-tokens: 생성할 최대 토큰 수
```

## 2. API 서버 실행 (LLMEngineAsync)

### vLLM API 서버 실행
```bash
# 기본 실행 방법
python -m vllm.entrypoints.api_server \
    --model [모델명] \
    --device fpga \
    --num-lpu-devices [LPU 개수] \
    --num-gpu-devices [GPU 개수]

# 예시
python -m vllm.entrypoints.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --device fpga \
    --num-lpu-devices 2 \
    --num-gpu-devices 0

# API 요청 테스트
python examples/lpu_client.py
```

### OpenAI 호환 API 서버 실행
```bash
# 기본 실행 방법
vllm serve [모델명] \
    --device fpga \
    --num-lpu-devices [LPU 개수] \
    --num-gpu-devices [GPU 개수] \
    --disable-frontend-multiprocessing

# 예시
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --device fpga \
    --num-lpu-devices 2 \
    --num-gpu-devices 0 \
    --disable-frontend-multiprocessing # vllm serve 사용 시 필수 옵션

# API 요청 테스트
python examples/lpu_openai_completion_client.py
```

## 3. 테스트벤치 실행

mini_testbench.sh 스크립트를 사용하여 다양한 설정으로 자동 테스트를 실행할 수 있습니다.

```bash
# 테스트벤치 실행
bash examples/mini_testbench.sh

# 테스트 설정 (mini_testbench.sh 내부 설정)
- 테스트 모델: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- LPU 디바이스 수: 1, 2
- GPU 디바이스 수: 0
- 각 설정당 요청 수: 3회

# 테스트 결과
- 결과는 log 디렉토리에 저장됨
- 타임스탬프 형식의 하위 디렉토리에 각 테스트 케이스별 로그 저장
- log/service_model_device.txt 파일에 전체 테스트 결과 요약
```


