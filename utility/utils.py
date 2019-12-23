import numpy as np
import argparse
import scipy.sparse as sp
import torch
from torch import nn
from dataloader.NewDatasets import AmazonBook, ML20M, Yelp2018, LastFM, ML20MContext, ML10M, App, Yelp, ML20MSparse
import psutil
from multiprocessing import Pool
from tqdm import tqdm
import os
#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--method', type=str, default='GMF',
                        help='choose the method.')

    parser.add_argument('-n', '--task', type=str, default='default',
                        help='task name.')

    parser.add_argument('--config', type=str, default='config.json',
                        help='config file.')

    parser.add_argument('--dataset', type=str, default='yelp', choices=['yelp', 'app', 'ml-20m-sparse'],
                        help='Choose a dataset from [\'yelp\', \'app\', \'ml-20m-sparse\'].')

    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')

    parser.add_argument('--topK', type=int, default=10,
                        help='topK for evaluation.')

    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')

    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')

    parser.add_argument('--multiplier', type=float, default=1.0,
                        help='multiplier over the similarity.')

    parser.add_argument('--bias', type=float, default=0.0,
                        help='bias over the similarity.')

    parser.add_argument('--loss', type=str, default='default',
                        help='Choose a loss type. [BCE, L2, CMP]')

    parser.add_argument('--reg', type=float, default=0.0,
                        help="L2 regularization of weight decay")

    parser.add_argument('--target', type=float, default=1.0,
                        help='soft target within (0.5, 1.0).')

    parser.add_argument('--bpr_margin', type=float, default=0.0,
                        help='margin for bpr. default=0.0')

    parser.add_argument('--mg_margin', type=float, default=0.1,
                        help='margin for bpr. default=0.0')

    # don't normalize user and item embedding by default
    parser.add_argument('--norm_user', action='store_true',
                        help='Whether to normalize user embedding.')

    parser.add_argument('--norm_item', action='store_true',
                        help='Whether to normalize item embedding.')

    parser.add_argument('--gmf_linear', action='store_true',
                        help='Whether to add a 1x1 fc before sigmoid in GMF.')

    parser.add_argument('--use_user_bias', action='store_true',
                        help='Whether to user bias.')

    parser.add_argument('--use_item_bias', action='store_true',
                        help='Whether to item bias.')

    parser.add_argument('--square_dist', action='store_true',
                        help='Whether to use the squared distance in CML.')

    parser.add_argument('--dist_norm', type=int, default=2,
                        help='Set the norm type (int) to use for DMF. default: 2 (l2).')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')

    parser.add_argument('--path_to_load_model', type=str, default='default',
                        help='Set the path to load the model to continue training.')

    parser.add_argument('--path_to_load_embedding', type=str, default='default',
                        help='Set the path to load the embedding to continue training.')

    parser.add_argument('--path_to_subclass_model', nargs=2, type=str, default=['default', 'default'],
                        help='Set the path to finetune the model for (NeuMF, DoubleGMF). [model_1, model_2]')

    parser.add_argument('--fix_left', action='store_true',
                        help='Whether to fix the left branch.')

    parser.add_argument('--fix_right', action='store_true',
                        help='Whether to fix the right branch.')

    parser.add_argument('--path_to_subclass_embedding', nargs=2, type=str, default=['default', 'default'],
                        help='Set the path to finetune the embedding for (NeuMF, DoubleGMF). [model_1, model_2]')

    parser.add_argument('--output', type=str, default='debug_output',
                        help='path to model outputs')

    parser.add_argument('--log_name', type=str, default='debug',
                        help='name of log file')

    parser.add_argument('--save_interval', type=int, default=10000,
                        help='interval for saving the trained model and info.')

    parser.add_argument('--gpu', type=int, default=-1,
                        help='gpu id')

    parser.add_argument('--item_pop', action='store_true',
                        help='item-pop evaluation, false for default')

    # scheduler parameters
    parser.add_argument('--eps', type=float, default=1000000,
                        help="eps for differential privacy")
    parser.add_argument('--cycle', type=int, default=-1,
                        help="cycle (episode) for distributed learning, -1 for centralized learning")

    parser.add_argument('--fuse', type=float, default=0.0,
                        help='fuse ratio of server side embeddings, 0.0 means private embeddings only')

    parser.add_argument('--regen', type=int, default=0,
                        help='0 for not regenerating fake data during each cycle')

    parser.add_argument('--thresh', type=float, default=0.5,
                        help='threshold for SVT report')

    # distributed learning on CPUs
    parser.add_argument('--mCore', action='store_true',
                        help='support for multi-cpu training')

    parser.add_argument('--world_size', type=int, default=4,
                        help='number of cores used in distributed learning')

    parser.add_argument("--local_rank", type=int, default=0)

    # item-based CF methods
    parser.add_argument("--num_neighbor", type=int, default=5,
                        help="num of similar item neighbors")

    parser.add_argument("--num_centroid", type=int, default=10,
                        help="num of centroids for clustering, 1 for no clustering")

    parser.add_argument("--clustering_method", type=str, default="kmeans",
                        help="method for clustering")

    parser.add_argument("--feature_type", type=str, default="word2vec",
                        help="type of features for clustering")

    parser.add_argument("--make_diary", action="store_false",
                        help="whether to create diary dirs")

    # normalization factor, used in KDD baseline
    parser.add_argument("--norm", type=str, default='l1', choices=['l1', 'l2'],
                        help="l1 or l2 normalization")

    parser.add_argument("--symm", type=int,
                        help="Symmetric flipping model or not")

    parser.add_argument("--sparsity", type=int, default=60,
                        help="sparsity for diff_sparsity datasets")

    args = parser.parse_args()

    if 'sparse' in args.dataset:
        args.dataset = args.dataset + '/{}'.format(args.sparsity)

    return args


def load_dataset(config, dataset, num_neg):
    if dataset == 'ml-20m':
        return ML20M(config['root_ml20m'], num_neg)
    elif dataset == 'last-fm':
        return LastFM(config['root_last-fm'], num_neg)
    elif dataset == 'yelp2018':
        return Yelp2018(config['root_yelp2018'], num_neg)
    elif dataset == 'amazon-book':
        return AmazonBook(config['root_amazon-book'], num_neg)
    elif dataset == 'ml-20m-context':
        return ML20MContext(config['root_ml20m_context'], num_neg)
    elif dataset == 'ml-10m':
        return ML10M(config['root_ml10m'], num_neg)
    elif dataset == 'yelp':
        return Yelp(config['root_yelp'], num_neg)
    elif dataset == 'app':
        return App(config['root_app'], num_neg)
    elif 'ml-20m-sparse/' in dataset:
        sparsity = int(dataset.split('/')[-1])
        return ML20MSparse(os.path.join(config['root_ml-sparse'], 'ml-s{}'.format(sparsity)), num_neg)
    else:
        raise (Exception('Dataset {0} not found'.format(dataset)))


#################### Training ####################

def adjust_lr(optimizer, lr_rate):
    group_num = len(optimizer.param_groups)
    for i in range(group_num):
        optimizer.param_groups[i]["lr"] = lr_rate


def save_model(dict_, path, model_name='default.pth'):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(dict_, os.path.join(path, model_name))


#################### File ####################
def load_rating_file_as_matrix(filename, splitter='\t'):
    '''
    Read .rating file and Return lil matrix.
    The first line of .rating file is: num_users\t num_items
    '''
    # Get number of users and items
    num_users, num_items = 0, 0
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.strip().split(splitter)
            u, i = int(arr[0]), int(arr[1])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
            line = f.readline()
    num_users, num_items = num_users + 1, num_items + 1

    # Construct matrix
    user_list, item_list = [], []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.strip().split(splitter)
            if len(arr) > 2:
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            else:
                user, item = int(arr[0]), int(arr[1])
                rating = 1

            if (rating > 0):
                user_list.append(user)
                item_list.append(item)
            line = f.readline()
    num = len(user_list)
    data = np.ones((num))
    user_list, item_list = np.array(user_list), np.array(item_list)
    mat = sp.coo_matrix((data, (user_list, item_list)), shape=(num_users, num_items))
    mat = mat.tolil()
    return mat


def load_rating_file_as_list(filename, splitter='\t'):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.strip().split(splitter)
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList


def load_negative_file(filename: str, splitter: str = ',') -> list:
    all_negative = []
    user_negative = []
    user_set = set()
    with open(filename, "r") as f:
        for line in f:
            arr = line.strip().split(splitter)
            user, item = int(arr[0]), int(arr[1])
            if user not in user_set:
                user_set.add(user)
                if user_negative:
                    all_negative.append(user_negative)
                    user_negative = []
            user_negative.append(item)
    all_negative.append(user_negative)
    return all_negative


#################### Model ####################
def make_fc_layers(cfg, in_channels=8):
    layers = []
    for v in cfg[:-1]:
        layers += [nn.Linear(in_channels, v), nn.ReLU(inplace=True)]
        in_channels = v
    layers += [nn.Linear(in_channels, cfg[-1])]
    return nn.Sequential(*layers)


def parallel_map(func, iter, parallel=True, worker_=-1, verbose=False):
    if verbose:
        iter = tqdm(iter)

    worker = psutil.cpu_count() if worker_ == -1 else worker_

    if parallel:
        with Pool(worker) as pool:
            results = pool.map(func, iter)
    else:
        results = list(map(func, iter))

    return results


def np_save(file_path, arr_dict: dict):
    dir_path = os.path.split(file_path)[0]
    print(dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if len(arr_dict) == 1:
        key, arr = list(arr_dict.items())[0]
        np.save(file_path, arr)
    else:
        np.savez(file_path, **arr_dict)