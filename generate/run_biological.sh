#!/bin/bash

INPUT_DIR="../data_example/biological_data/FGT_data"
GPU_ID=0  


mapfile -t TASKS < <(find "$INPUT_DIR" -maxdepth 1 -name "FGT_*.json" | sort)

for current_task in "${TASKS[@]}"; do
    echo "[INFO] 开始处理任务 '${current_task}' on GPU $GPU_ID"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python biological_successRate.py "$current_task"
    
    echo "[INFO] 完成任务 '${current_task}'"
done

echo "[INFO] 所有任务已完成。"
