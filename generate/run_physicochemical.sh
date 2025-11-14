#!/bin/bash

types=("LogP" "SA" "Esol" "TPSA")
for type in ${types[@]}
do
    CUDA_VISIBLE_DEVICES=3 python physicochemical_successRate.py ../data_example/physicochemical_data/FGT_data/FGT_${type}_test.json
done
