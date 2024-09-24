
log_sum="log/service_model_device.txt"

model_ids=("TinyLlama/TinyLlama-1.1B-Chat-v1.0") # "facebook/opt-1.3b" "huggyllama/llama-7b")
num_lpu_devices=(2) #4
num_gpu_devices=(0)

current_datetime=$(date "+%Y-%m-%d %H:%M:%S")
echo "$current_datetime"
echo "$current_datetime" >> ${log_sum}

# LLMEngine Test
for model_id in "${model_ids[@]}"; do
  for num_lpu_device in "${num_lpu_devices[@]}"; do
   for num_gpu_device in "${num_gpu_devices[@]}"; do
    #IFS='\' read -ra parts <<< "$model_id"
    #model_name="${parts[-1]}"
    model_name=$(echo "$model_id" | awk -F'/' '{print $NF}')
    echo "*********************************"
    echo "**** Start inference_${model_name}_${num_lpu_device}_${num_gpu_device}"
    echo "*********************************"
    python lpu_inference_arg.py -m ${model_id} -l ${num_lpu_device} -g ${num_gpu_device} > log/inference_${model_name}_${num_lpu_device}_${num_gpu_device}.txt
    echo "*********************************" >> ${log_sum}
    echo "[Testbench] The Result of log/inference_${model_name}_${num_lpu_device}_${num_gpu_device}.txt" >> ${log_sum}
    tail -n 1 "log/inference_${model_name}_${num_lpu_device}_${num_gpu_device}.txt" >> ${log_sum}
    echo "" >> ${log_sum}
   done
  done
done

# LLMEngineAsync Test with vLLM serve
for model_id in "${model_ids[@]}"; do
  for num_lpu_device in "${num_lpu_devices[@]}"; do
   for num_gpu_device in "${num_gpu_devices[@]}"; do
    model_name=$(echo "$model_id" | awk -F'/' '{print $NF}')
    echo "*********************************"
    echo "**** Start serving_${model_name}_${num_lpu_device}_${num_gpu_device}"
    echo "*********************************"
    python -m vllm.entrypoints.api_server --model ${model_id} --device fpga --num-lpu-devices ${num_lpu_device} --num-gpu-devices ${num_gpu_device} &

    # Waiting for server
    while ! nc -z localhost "8000"; do  
        echo "[Testbench] Waiting for server..."
        sleep 3 
    done
    echo "[Testbench] The server is ready!"

    python lpu_client.py > log/vllm_serve_${model_name}_${num_lpu_device}_${num_gpu_device}.txt

    # Waiting for process kill
    PID=$(jobs -p | tail -n 1)
    if [ -n "$PID" ]; then
        kill -SIGINT "$PID"
        while true; do
            if ps -p "$PID" > /dev/null; then
                echo "[Testbench] Kill the process..."
                sleep 3
            else
                echo "[Testbench] Process (PID: $PID) is killed."
                break
            fi
        done
    fi

    # Write log in text file
    echo "*********************************" >> ${log_sum}
    echo "The Result of log/vllm_serve_${model_name}_${num_lpu_device}_${num_gpu_device}.txt" >> ${log_sum}
    tail -n 1 "log/vllm_serve_${model_name}_${num_lpu_device}_${num_gpu_device}.txt" >> ${log_sum}
    echo "" >> ${log_sum}
   done
  done
done



# OpenAI API Test
model_id=${model_ids[0]}
num_lpu_device=${num_lpu_devices[0]}
num_gpu_device=${num_gpu_devices[0]}
model_name=$(echo "$model_id" | awk -F'/' '{print $NF}')
echo "*********************************"
echo "**** Start serving_${model_name}_${num_lpu_device}_${num_gpu_device}"
echo "*********************************"
python -m vllm.entrypoints.openai.api_server --model ${model_id} --device fpga --num-lpu-devices ${num_lpu_device} --num_gpu_devices ${num_gpu_device} &

# Waiting for server
while ! nc -z localhost "8000"; do  
    echo "[Testbench] Waiting for server..."
    sleep 3 
done
echo "[Testbench] The server is ready!"

python lpu_openai_completion_client.py > log/openai_serve_${model_name}_${num_lpu_device}_${num_gpu_device}.txt

# Waiting for process kill
PID=$(jobs -p | tail -n 1)
if [ -n "$PID" ]; then
    kill -SIGINT "$PID"
    while true; do
        if ps -p "$PID" > /dev/null; then
            echo "[Testbench] Kill the process..."
            sleep 3
        else
            echo "[Testbench] Process (PID: $PID) is killed."
            break
        fi
    done
fi

# Write log in text file
echo "*********************************" >> ${log_sum}
echo "The Result of log/openai_serve_${model_name}_${num_lpu_device}_${num_gpu_device}.txt" >> ${log_sum}
tail -n 1 "log/openai_serve_${model_name}_${num_lpu_device}_${num_gpu_device}.txt" >> ${log_sum}
echo "" >> ${log_sum}
