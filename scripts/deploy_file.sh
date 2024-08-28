
site_packages=$(python3 -c "import site; print(site.getsitepackages()[0])")

cp _C.abi3.so ${site_packages}/vllm/.
cp commit_id.py ${site_packages}/vllm/.
cp _core_C.abi3.so ${site_packages}/vllm/.
cp _moe_C.abi3.so ${site_packages}/vllm/.
