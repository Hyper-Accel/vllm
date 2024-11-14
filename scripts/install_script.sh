
site_packages=$(python3 -c "import site; print(site.getsitepackages()[0])")

pip install numpy==1.26.0
pip install vllm==0.5.5
pip install mistral_common
echo "Start move vllm to ${site_packages}"
mv ${site_packages}/vllm ${site_packages}/vllm_bk
echo "Done backup"
cp -r ../vllm ${site_packages}/vllm
echo "Done copy"

cp ${site_packages}/vllm_bk/_C.abi3.so ${site_packages}/vllm/.
cp ${site_packages}/vllm_bk/_core_C.abi3.so ${site_packages}/vllm/.
cp ${site_packages}/vllm_bk/_moe_C.abi3.so ${site_packages}/vllm/.

