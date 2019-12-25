# Distributed Private Recsys

This is the implementation of locally differentially private distributed recommender system.

The basic idea of this model is constructing an Item-KNN model with local perturbation and central parameter inference. In order to satisfy the condition of differential privacy, randomized response is applied on each client to preserve user privacy.

In order to evaluate the performance of our model, we also introduce three types of baselines:

- Matrix Factorization models: Generalized Matrix Factorization (GMF), Matrix Factorization (MF). Both of them directly utilize noisy data after local perturbation without a further step of parameter inference. 
- Noisy ItemKNN: This KNN model directly utilize noisy data without any post-processing or estimation.
- Optimal: ItemKNN using original data. This model is supposed to have the best performance but it also lacks privacy protection. 

## Environment Settings
We use PyTorch as the backend for models using GPU (MF, GMF) and Python3 for models using CPU. Specific settings can be found in [requirements.txt](https://github.com/HenryNebula/Distributed_Private_Recsys/blob/master/requirements.txt).

## Example to run the codes.
The instruction of commands has been clearly stated in the codes (see the  utility.utils.parse_args function). 

Run DPKNN-like models:
```bash
python3 sim_based_train.py \
                        --task DPKNN \
                        --gpu -1 \
                        --dataset yelp \
                        --method itemCF \
                        --config itemsim_config.json \
                        --topK 5 \
                        --num_neighbor 10 \
                        --eps 1.0 
```

Run MF-like models:
```bash
python3 mf_based_train.py \
                    --task mf-train \
                    --gpu 0 \
                    --dataset yelp \
                    --method MF \
                    --config itemsim_config.json \
                    --topK 5 \
                    --lr 0.001 \
                    --reg 0.001 \
                    --num_factors 16 \
                    --eps 1.0 \
                    --symm 1 \
                    --loss BCE \
                    --epochs 200
```

### Dataset
We provide three processed datasets: [MovieLens 20 Million](http://files.grouplens.org/datasets/movielens/ml-20m.zip) (called ml-sparse/ml-s60 in 
[data](https://github.com/HenryNebula/Distributed_Private_Recsys/tree/master/data) folder), [Yelp 2018](https://www.yelp.com/dataset) (called Yelp in [data](https://github.com/HenryNebula/Distributed_Private_Recsys/tree/master/data) folder) and [AppUsage](http://www.shazhao.net/applens2019/) 
(called app in [data](https://github.com/HenryNebula/Distributed_Private_Recsys/tree/master/data) folder). All datasets are split to train, validation and test parts, using leave-one-out strategy.

Basic statistics of three datasets are listed below,

| Dataset | #User | #Item | #Sparsity |
| :-----: | :------: | :------: | :------: |
| AppUsage | 871 | 1682 |96.455%
| MovieLens | 74529 | 9953 | 97.459%
|Yelp | 6829| 9132| 99.322%|

Here is a short description of each file. Note that in order to load the dataset properly, try the following code,
```bash
cd data/
tar -xvzf dataset.tar.gz
```

train.dat: 
- Train file.
- Each Line is a training instance: userID,itemID (both starting from 0)

test.dat:
- Test file (positive instances, every user has **1** positive instance). 
- Each Line is a testing instance: userID,itemID (both starting from 0)

test.negative.dat
- Test file (negative instances, every user has **99** negative instances).
- Each line is in the format: userID,itemID (both starting from 0)

For validation set, the setting and format are the same with test set.

### Acknowledgement

The implementation of Dataloader and MF-like models is inspired by Shaohui Liu ([B1ueber2y](https://github.com/B1ueber2y))"s implementation of NCF (PyTorch Version). Thanks a lot!