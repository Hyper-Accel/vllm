
site_packages=$(python3 -c "import site; print(site.getsitepackages()[0])")

cp ${site_packages}/vllm/_C.abi3.so .
cp ${site_packages}/vllm/commit_id.py .
cp ${site_packages}/vllm/_core_C.abi3.so .
cp ${site_packages}/vllm/_moe_C.abi3.so .
