#!/bin/bash

# 获取 'target_data' 目录下的所有文件
files=./target_data/*

# 遍历所有文件
for file in $files
do
   # 运行 Python 脚本，将文件路径作为第一个参数，1 作为第二个参数
   python ./script/change_instruction.py $file 1
done
