
log_sum="log/service_model_device.txt"

model_ids=("TinyLlama/TinyLlama-1.1B-Chat-v1.0") # "facebook/opt-1.3b" "huggyllama/llama-7b")
num_devices=(1 2 4) 

current_datetime=$(date "+%Y-%m-%d %H:%M:%S")
echo "$current_datetime"
echo "$current_datetime" >> ${log_sum}

"""
for model_id in "${model_ids[@]}"; do
  for num_device in "${num_devices[@]}"; do
    #IFS='\' read -ra parts <<< "$model_id"
    #model_name="${parts[-1]}"
    model_name=$(echo "$model_id" | awk -F'/' '{print $NF}')
    echo "*********************************"
    echo "**** Start inference_${model_name}_${num_device}"
    echo "*********************************"
    python lpu_inference_arg.py -m ${model_id} -n ${num_device} > log/inference_${model_name}_${num_device}.txt
    echo "*********************************" >> ${log_sum}
    echo "The Result of log/inference_${model_name}_${num_device}.txt" >> ${log_sum}
    tail -n 1 "log/inference_${model_name}_${num_device}.txt" >> ${log_sum}
    echo "" >> ${log_sum}
  done
done
"""

for model_id in "${model_ids[@]}"; do
  for num_device in "${num_devices[@]}"; do
    model_name=$(echo "$model_id" | awk -F'/' '{print $NF}')
    echo "*********************************"
    echo "**** Start serving_${model_name}_${num_device}"
    echo "*********************************"
    python -m vllm.entrypoints.api_server --model ${model_id} --device fpga --tensor-parallel-size ${num_device} &

    # Waiting for server
    while ! nc -z localhost "8000"; do  
        echo "Waiting for server..."
        sleep 3 
    done
    echo "The server is ready!"

    python lpu_client.py > log/vllm_serve_${model_name}_${num_device}.txt

    # Waiting for process kill
    PID=$(jobs -p | tail -n 1)
    if [ -n "$PID" ]; then
        kill -SIGINT "$PID"
        while true; do
            if ps -p "$PID" > /dev/null; then
                echo "Kill the process..."
                sleep 3
            else
                echo "Process (PID: $PID) is killed."
                break
            fi
        done
    fi

    # Write log in text file
    echo "*********************************" >> ${log_sum}
    echo "The Result of log/vllm_serve_${model_name}_${num_device}.txt" >> ${log_sum}
    tail -n 1 "log/vllm_serve_${model_name}_${num_device}.txt" >> ${log_sum}
    echo "" >> ${log_sum}
  done
done


