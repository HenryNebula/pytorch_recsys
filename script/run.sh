#!/bin/bash

gpu="0"

#Dataset
dataset="ml-1m"

method_list=("MF")

config="default_config.json"

topK="5"

lr_list=("0.3")

reg_list=("0")

embed_list=("8")

cd ../

python="/home/henryhuang/.conda/envs/recsys/bin/python"

# shellcheck disable=SC2068
for method in ${method_list[@]} ; do
    for lr in ${lr_list[@]} ; do
        for reg in ${reg_list[@]} ; do
            for embed in ${embed_list[@]} ; do
                  description=${method}-lr${lr}-reg${reg}-embed${embed}
                  procname=${description}

                  ${python} mf_based_train.py \
                  --task ${procname} \
                  --gpu ${gpu} \
                  --dataset ${dataset} \
                  --method ${method} \
                  --config ${config} \
                  --topK ${topK} \
                  --lr ${lr} \
                  --reg ${reg} \
                  --num_factors ${embed} \
                  --loss BCE \
                  --epochs 200 \
                  --batch_size -1 \
                  --implicit 1
            done
        done
    done
done