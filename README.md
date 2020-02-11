# PyTorch Recommender Systems

This is the implementation of several popular recommender system algorithms with both explicit and implicit data input. For implicit dataset, a fast negative sampling procedure is implemented with the help of Numba. Currently, Matrix Factorization (MF), Generalized MF (GMF) and Multi-layer Perceptron (MLP) have been implemented and tested.
 
## Environment Settings
We use PyTorch as the backend for all models using GPU (MF, GMF). Specific settings can be found in [requirements.txt](https://github.com/HenryNebula/Distributed_Private_Recsys/blob/master/requirements.txt).

## Example to run the codes.
The instruction of commands has been clearly stated in the codes (see the  utility.utils.parse_args function). 

Run MF-like models:
```bash
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
```

### Dataset
Currently, only a pre-processed [MovieLens 1 Million](https://console.cloud.google.com/storage/browser/recsys101-bucket/datasets) dataset has been tested with this implementation. The preprocessing includes generating negative samples for the test ground truth and changing original dataset to an implicit form.

train.dat: 
- Train file.
- Each Line is a training instance: userID,itemID (both starting from 0)

test.dat:
- Test file (positive instances, every user has **1** positive instance). 
- Each Line is a testing instance: userID,itemID (both starting from 0)

test.negative.dat
- Test file (negative instances, every user has **99** negative instances).
- Each line is in the format: userID,itemID (both starting from 0)

### Acknowledgement

The implementation of Dataloader and MF-like models is inspired by Shaohui Liu ([B1ueber2y](https://github.com/B1ueber2y))"s implementation of NCF (PyTorch Version). Thanks a lot!