from Trainer import Trainer
from noise.SVT import report_with_SVT
import numpy as np
import torch
from torch.autograd import Variable
from utility.fast_rank_topK import fast_topK
from scipy.sparse import lil_matrix
import psutil
import utility.utils as utils
import os
import torch.distributed as dist
import time
from tqdm import tqdm


def gen_with_predictions(user, trainer):
    items = list(range(trainer.dataset.num_items))
    users_np = np.full(len(items), user, dtype='int32')
    items_np = np.array(items)
    users_tensor = Variable(torch.from_numpy(users_np).to(trainer.model.device)).long()
    items_tensor = Variable(torch.from_numpy(items_np).to(trainer.model.device)).long()
    predictions_gpu = trainer.model.get_predictions([users_tensor, items_tensor])
    predictions = predictions_gpu.cpu().data.numpy().reshape(-1)
    topK = int(trainer.dataset.train_data[user, :].sum())

    ranklist = fast_topK(predictions, topK)
    del predictions,predictions_gpu,users_tensor, items_tensor
    return ranklist


def val_set_generator(user_tup):
    # user_tup: (user, ranklist, item_num, embed_flag)
    user, ranklist, item_num, embed_flag = user_tup
    if len(ranklist) > 1:
        neg_set = set(range(item_num)).difference(set(ranklist))
        val_neg = np.random.choice(list(neg_set), 99, replace=False).tolist()
        # add val_pos:[user*1, item*1]
        val_pos = ranklist[-1]
        val_pos = [user, val_pos]
        gt = ranklist[:-1]
    else:
        gt = ranklist
        val_pos = []
        val_neg = []
    results = [user, gt, val_pos, val_neg]
    return results


def direct_report_wrapper(user_tup):
    user, user_row, num_item, report_noise = user_tup
    # report_noise: -1 for center (direct_report), 0 for clean dist (user_row is index already)
    # other value is eps

    if report_noise == -1:
        idx_row = user_row
    else:
        thresh = 0.05
        idx_row = report_with_SVT(user_row, report_noise, num_item, thresh)
    return (user, idx_row, num_item, False)


def collect_data(trainer:Trainer, use_embed=False):
    train_data = lil_matrix((trainer.dataset.num_users, trainer.dataset.num_items))
    new_data = {'val_pos':[], 'val_neg':[], 'train_data': None}

    # avoid causing cpu too busy
    while psutil.cpu_percent() > 90.0:
        sleep_time = np.random.randint(30, 60)
        print("CPU too busy:{0}, sleeping {1:d} seconds".format(psutil.cpu_percent(), sleep_time))
        time.sleep(sleep_time)

    num_item = trainer.dataset.num_items
    report_noise = -1 if trainer.args.eps > 10000 else trainer.args.eps

    if use_embed:
        user_tup = map(lambda x: (x, gen_with_predictions(x, trainer), num_item, report_noise), tqdm(range(trainer.dataset.num_users)))
    else:
        user_tup = map(lambda x: (x, trainer.dataset.train_data[x, :].rows[0], num_item, report_noise), tqdm(range(trainer.dataset.num_users)))

    trainer.logger.info("Finish collecting raw report data")

    results = utils.parallel_map(direct_report_wrapper, tqdm(list(user_tup)), parallel=True)
    trainer.logger.info("Finish reporting data after obfuscation")

    results = utils.parallel_map(val_set_generator, tqdm(results), parallel=True)
    trainer.logger.info("Finish generating negative samples for val set")

    for result in results:
        user, gt, val_pos, val_neg = result
        if val_pos and val_neg:
            new_data['val_pos'].append(val_pos)
            new_data['val_neg'].append(val_neg)
        train_data[user, gt] = 1
    new_data['train_data'] = train_data

    return new_data


def init_processes(rank, args, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    size = args.world_size
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)