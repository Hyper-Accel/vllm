
site_packages=$(python3 -c "import site; print(site.getsitepackages()[0])")

pip install numpy==1.26.0
pip install vllm==0.5.5
pip install mistral_common
echo "Start move vllm to ${site_packages}"
sudo mv ${site_packages}/vllm ${site_packages}/vllm_bk
echo "Done backup"
sudo cp -r ../vllm ${site_packages}/vllm
echo "Done copy"
