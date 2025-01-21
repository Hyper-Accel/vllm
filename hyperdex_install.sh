pip install -i https://hyperaccel:*hyper123@pypi.hyperaccel.ai/simple/ hyperdex-transformers==0.0.2+cpu
pip install -i https://hyperaccel:*hyper123@pypi.hyperaccel.ai/simple/ hyperdex-compiler==0.0.1+cpu

python use_existing_torch.py

pip install -r requirements-build.txt

# sudo apt-get update  -y
# sudo apt-get install -y gcc-12 g++-12 libnuma-dev
# sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10
# sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 10


# sudo cmake -B ../oneDNN/build -S ../oneDNN -G Ninja -DONEDNN_LIBRARY_TYPE=STATIC \
#     -DONEDNN_BUILD_DOC=OFF \
#     -DONEDNN_BUILD_EXAMPLES=OFF \
#     -DONEDNN_BUILD_TESTS=OFF \
#     -DONEDNN_BUILD_GRAPH=OFF \
#     -DONEDNN_ENABLE_WORKLOAD=INFERENCE \
#     -DONEDNN_ENABLE_PRIMITIVE=MATMUL

# sudo cmake --build ../oneDNN/build --target install --config Release
VLLM_TARGET_DEVICE=fpga pip install -e . --no-build-isolation

