#!/bin/bash

if [[ $HOSTNAME = fib-dl ]]; then
    gpu="7"
else
    gpu="-1"
fi

## Dataset
#dataset="ml-20m-context"
#dataset="ml-10m"
#dataset="app"
dataset="yelp"
#dataset="ml-20m-sparse"
sparsity="60"

method_list=("GMF")
symm=1

# fix paras
config="itemsim_config.json"
topK="5"

#eps_list=("0.6" "0.4")
eps_list=("1")
#eps_list=("0.6" "0.7" "0.9" "1.0")
#eps_list=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
#eps_list=("1.1" "1.2" "1.3" "1.4" "1.5" "1.6" "1.7" "1.8" "1.9" "2.0")
#eps_list=( "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0" "1.1" "1.2" "1.3" "1.4" "1.5" "1.6" "1.7" "1.8" "1.9" "2.0")

lr_list=("0.0001" "0.0005")
#lr_list=("0.0001")

#reg_list=("0.0001")
reg_list=("0" "0.0005")

embed_list=("8" "16")
#embed_list=("8")

cycle=-1

cd ../

if [[ ${gpu} = -1 ]]; then
    python="/home/fib/anaconda3/bin/python"
else
    python="python3"
fi

for method in ${method_list[@]} ; do
    for lr in ${lr_list[@]} ; do
        for reg in ${reg_list[@]} ; do
            for embed in ${embed_list[@]} ; do
                for eps in ${eps_list[@]} ; do

                    description=${method}-eps${eps}-lr${lr}-reg${reg}-embed${embed}
                    procname=${description}

                    ${python} mf_based_train.py \
                    --task ${procname}@gaochen \
                    --gpu ${gpu} \
                    --dataset ${dataset} \
                    --method ${method} \
                    --config ${config} \
                    --topK ${topK} \
                    --lr ${lr} \
                    --reg ${reg} \
                    --num_factors ${embed} \
                    --eps ${eps} \
                    --cycle ${cycle} \
                    --symm ${symm} \
                    --loss BCE \
                    --epochs 200 \
                    --sparsity ${sparsity} &
                done
            done
        done
    done
done