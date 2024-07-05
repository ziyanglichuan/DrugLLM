#!/bin/bash

# 输入和输出目录
input_dir="./changeInstru_data/target"
output_dir="./target_encodeData"

# 遍历输入目录中的所有文件
for filename in "$input_dir"/*; do
    # 获取文件名（不包括路径）
    base_filename=$(basename "$filename")
    # 使用Python脚本处理文件
    python ./script/encode_jsonSmiles.py "$base_filename" "$input_dir" "$output_dir"
done

