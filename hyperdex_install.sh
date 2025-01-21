#!/bin/bash

# usage: ./hyperdex_install.sh <pypi username> <pypi password>

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./hyperdex_install.sh <pypi username> <pypi password>"
    exit 1
fi

pip install -i https://$1:$2@pypi.hyperaccel.ai/simple/ hyperdex-transformers==0.0.2+cpu
pip install -i https://$1:$2@pypi.hyperaccel.ai/simple/ hyperdex-compiler==0.0.1+cpu

python use_existing_torch.py

pip install -r requirements-build.txt

VLLM_TARGET_DEVICE=fpga pip install -e . --no-build-isolation

