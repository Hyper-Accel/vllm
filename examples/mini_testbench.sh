#!/bin/bash
sudo apt-get install -y netcat
set -e

log_sum="log/service_model_device.txt"

model_ids=("TinyLlama/TinyLlama-1.1B-Chat-v1.0") # "facebook/opt-1.3b" "huggyllama/llama-7b")
num_lpu_devices=(1 2) #4
num_gpu_devices=(0)
num_requests=3

current_datetime=$(date "+%Y-%m-%d_%H-%M-%S")
mkdir -p log/${current_datetime}
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
    for i in $(seq 1 $num_requests); do
        if [ $i -eq 1 ]; then
            python lpu_inference_arg.py -m ${model_id} -l ${num_lpu_device} -g ${num_gpu_device} > log/${current_datetime}/inference_${model_name}_${num_lpu_device}_${num_gpu_device}.txt
        else
            python lpu_inference_arg.py -m ${model_id} -l ${num_lpu_device} -g ${num_gpu_device} >> log/${current_datetime}/inference_${model_name}_${num_lpu_device}_${num_gpu_device}.txt
        fi
        
        # Check if the python script failed
        if [ $? -ne 0 ]; then
            echo "Error: Python script failed"
            exit 1
        fi
    done
    echo "*********************************" >> ${log_sum}
    echo "[Testbench] The Result of log/${current_datetime}/inference_${model_name}_${num_lpu_device}_${num_gpu_device}.txt" >> ${log_sum}
    tail -n 1 "log/${current_datetime}/inference_${model_name}_${num_lpu_device}_${num_gpu_device}.txt" >> ${log_sum}
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

    # Waiting for server with timeout (3 minutes)
    wait_count=0
    max_wait=60  # 3분 (3초 * 60)
    while ! nc -z localhost "8000"; do  
        echo "[Testbench] Waiting for server... (${wait_count}/60)"
        sleep 3
        wait_count=$((wait_count + 1))
        if [ $wait_count -ge $max_wait ]; then
            echo "Error: Server did not start within 3 minutes"
            # Kill the server process if it exists
            PID=$(jobs -p | tail -n 1)
            if [ -n "$PID" ]; then
                kill -SIGINT "$PID"
            fi
            exit 1
        fi
    done
    echo "[Testbench] The server is ready!"

    for i in $(seq 1 $num_requests); do
        if [ $i -eq 1 ]; then
            python lpu_client.py > log/${current_datetime}/vllm_serve_${model_name}_${num_lpu_device}_${num_gpu_device}.txt
        else
            python lpu_client.py >> log/${current_datetime}/vllm_serve_${model_name}_${num_lpu_device}_${num_gpu_device}.txt
        fi
        
        if [ $? -ne 0 ]; then
            echo "Error: Python client script failed"
            # Kill the server process if it exists
            PID=$(jobs -p | tail -n 1)
            if [ -n "$PID" ]; then
                kill -SIGINT "$PID"
            fi
            exit 1
        fi
    done

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
    echo "The Result of log/${current_datetime}/vllm_serve_${model_name}_${num_lpu_device}_${num_gpu_device}.txt" >> ${log_sum}
    tail -n 1 "log/${current_datetime}/vllm_serve_${model_name}_${num_lpu_device}_${num_gpu_device}.txt" >> ${log_sum}
    echo "" >> ${log_sum}
   done
  done
done



# OpenAI API Test
for model_id in "${model_ids[@]}"; do
  for num_lpu_device in "${num_lpu_devices[@]}"; do
   for num_gpu_device in "${num_gpu_devices[@]}"; do
    model_name=$(echo "$model_id" | awk -F'/' '{print $NF}')
    echo "*********************************"
    echo "**** Start serving_${model_name}_${num_lpu_device}_${num_gpu_device}"
    echo "*********************************"
    python -m vllm.entrypoints.openai.api_server --model ${model_id} --device fpga --num-lpu-devices ${num_lpu_device} --num_gpu_devices ${num_gpu_device} &

    # Waiting for server with timeout (3 minutes)
    wait_count=0
    max_wait=60  # 3분 (3초 * 60)
    while ! nc -z localhost "8000"; do  
        echo "[Testbench] Waiting for server... (${wait_count}/60)"
        sleep 3
        wait_count=$((wait_count + 1))
        if [ $wait_count -ge $max_wait ]; then
            echo "Error: Server did not start within 3 minutes"
            # Kill the server process if it exists
            PID=$(jobs -p | tail -n 1)
            if [ -n "$PID" ]; then
                kill -SIGINT "$PID"
            fi
            exit 1
        fi
    done
    echo "[Testbench] The server is ready!"

    for i in $(seq 1 $num_requests); do
        if [ $i -eq 1 ]; then
            python lpu_openai_completion_client.py > log/${current_datetime}/openai_serve_${model_name}_${num_lpu_device}_${num_gpu_device}.txt
        else
            python lpu_openai_completion_client.py >> log/${current_datetime}/openai_serve_${model_name}_${num_lpu_device}_${num_gpu_device}.txt
        fi
        
        if [ $? -ne 0 ]; then
            echo "Error: Python client script failed"
            # Kill the server process if it exists
            PID=$(jobs -p | tail -n 1)
            if [ -n "$PID" ]; then
                kill -SIGINT "$PID"
            fi
            exit 1
        fi
    done

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
   done
  done
done
# Write log in text file
echo "*********************************" >> ${log_sum}
echo "The Result of log/${current_datetime}/openai_serve_${model_name}_${num_lpu_device}_${num_gpu_device}.txt" >> ${log_sum}
tail -n 1 "log/${current_datetime}/openai_serve_${model_name}_${num_lpu_device}_${num_gpu_device}.txt" >> ${log_sum}
echo "" >> ${log_sum}
