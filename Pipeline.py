from torch import optim
import torch
from torch.utils.data import DataLoader
import utility.utils as utils
from utility.diary import Diary
from utility.logger import create_logger

from model.MF import MF
from model.GMF import GMF
from model.NeuMF import NeuMF
from model.CML import CML

import json
import os
import multiprocessing
import numpy as np


class Pipeline:
    def __init__(self, args):

        self.args = args

        # load logger
        self.logger = create_logger(os.path.join(args.output, args.log_name))

        # load generic config
        with open(os.path.join(os.path.dirname(__file__), "config_json", args.config)) as f:
            self.config = json.loads(f.read())

        self.logger.debug("start loading dataset!")

        # dataset abstraction
        self.dataset = utils.load_dataset(self.config, args.dataset, args.num_neg, args.implicit)
        self.train_dataloader = self.get_data_loader()

        self.evaluator = self.dataset.get_evaluator()

        self.logger.debug("finish loading evaluators for both val and test phase")

        gpu_index = self.args.gpu

        assert gpu_index <= torch.cuda.device_count() - 1, "Wrong GPU number {}".format(gpu_index)

        self.device = torch.device("cpu") if gpu_index < 0 else torch.device("cuda: {}".format(gpu_index))

        # model parameters
        self.model, self.optimizer, self.lr = self.load_model()

    def get_data_loader(self):
        # mini-batch or full-batch depends on the size of dataset
        batch_size = self.args.batch_size
        if batch_size > 0:
            return DataLoader(self.dataset, batch_size=batch_size,
                              shuffle=self.config["shuffle"],
                              num_workers=multiprocessing.cpu_count(),
                              drop_last=True)

        else:
            return (self.update_full_batch_with_neg_samples() for _ in range(self.args.epochs))

    def update_full_batch_with_neg_samples(self):
        num_neg = self.args.num_neg

        users_pos = self.dataset.train_ratings.row
        items_pos = self.dataset.train_ratings.col
        labels_pos = self.dataset.train_ratings.data

        num_pos = len(users_pos)

        users = np.tile(np.array(users_pos, dtype=np.int64), (num_neg + 1))
        labels = np.hstack([labels_pos, np.zeros((num_pos * num_neg,), dtype=np.float)])

        neg_items = [] if num_neg <= 0 else \
            [self.dataset.find_neg(self.dataset.train_rel_items[usr], self.dataset.num_items, num_neg)
             for usr in users_pos]

        items = np.hstack([np.array(items_pos, dtype=np.int64), np.array(neg_items, dtype=np.int64).reshape(-1)])

        # full dataset into GPU
        users = torch.from_numpy(users)
        items = torch.from_numpy(items)
        labels = torch.from_numpy(labels)

        return users, items, labels

    def load_model(self):
        # Model
        args = self.args

        num_users, num_items = self.dataset.num_users, self.dataset.num_items

        dict_config = {"num_factors": args.num_factors,
                       "num_users": num_users, "num_items": num_items,
                       "implicit": args.implicit,
                       "num_neg": args.num_neg,
                       "loss_type": args.loss,
                       "reg": args.reg,
                       "device": self.device,
                       "norm_user": args.norm_user, "norm_item": args.norm_item,
                       "use_user_bias": args.use_user_bias, "use_item_bias": args.use_item_bias,
                       "multiplier": args.multiplier, "bias": args.bias,
                       "square_dist": args.square_dist}

        model_class_dict = {
            "MF": MF,
            "GMF": GMF,
            "NeuMF": NeuMF,
            "CML": CML
        }
        if args.method in model_class_dict:
            model = model_class_dict[args.method](dict_config)

        else:
            raise (Exception("Method {0} not found. Choose the method from {1}".format(args.method, model_class_dict.keys())))

        model.to(self.device)

        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.99), weight_decay=args.reg)

        return model, optimizer, lr

