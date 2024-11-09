#!/bin/bash

types=("SA" "LogP" "Esol" "TPSA")
for type in ${types[@]}
do
    CUDA_VISIBLE_DEVICES=0 python GMR_genarate.py ../data_example/GMR_data/GMR_${type}_test.json
done
