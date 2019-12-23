#!/bin/bash

gpu="-1"

## Dataset
#dataset="ml-20m-context"
#dataset="ml-10m"
#dataset="app"
#dataset="yelp"
dataset="ml-20m-sparse"
sparsity="60"


method_list=("ItemCF")
#method_list=("ItemCF_Optimal")
#method_list=("ItemCF_KDD")
#method_list=("ItemCF_KDD" "ItemCF_Noisy" "ItemCF_Noisy_Optimal")

# fix paras
config="itemsim_config.json"
topK="5"

#neighbor_list=("20" "30" "40" "50" "60" "70" "80" "90")
neighbor_list=("10")

num_centroid_list=("1")

#eps_list=("0.6" "0.4")
eps_list=("1")
#eps_list=("0.6" "0.7" "0.9" "1.0")
#eps_list=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
#eps_list=("1.1" "1.2" "1.3" "1.4" "1.5" "1.6" "1.7" "1.8" "1.9" "2.0")
#eps_list=( "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0" "1.1" "1.2" "1.3" "1.4" "1.5" "1.6" "1.7" "1.8" "1.9" "2.0")

cluster_method_list=("kmeans")
feature_type_list=("word2vec")
norm="l2"
cycle=1

cd ../
# :<<BLOCK
for feature_type in ${feature_type_list[@]} ; do
    for cluster_method in ${cluster_method_list[@]} ; do
        for num_centroid in ${num_centroid_list[@]} ; do
            for eps in ${eps_list[@]} ; do
                for method in ${method_list[@]} ; do
                    for num_neighbor in ${neighbor_list[@]} ; do

                        description=${method}-eps${eps}-${cluster_method}-${feature_type}
                        procname=${description}

                        /home/fib/anaconda3/bin/python3 sim_based_train.py \
                        --task ${procname}@c-huang \
                        --gpu ${gpu} \
                        --dataset ${dataset} \
                        --method ${method} \
                        --config ${config} \
                        --topK ${topK} \
                        --num_neighbor ${num_neighbor} \
                        --num_centroid ${num_centroid} \
                        --clustering_method ${cluster_method} \
                        --feature_type ${feature_type} \
                        --eps ${eps} \
                        --cycle ${cycle} \
                        --norm ${norm} \
                        --sparsity ${sparsity}
                    done
                done
            done
        done
    done
done
# BLOCK