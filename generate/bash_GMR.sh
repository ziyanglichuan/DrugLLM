#!/bin/bash

types=("SA" "logp" "esol" "TPSA")
# 调用脚本
for i in {0..9}
do
    for type in ${types[@]}
    do
	    CUDA_VISIBLE_DEVICES=0 python GMR_genarate.py ../data/encode_data/encode_change2_offset${i}_${type}.json
    done
done
