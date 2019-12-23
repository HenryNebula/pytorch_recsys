import os, sys
from dataloader.AbstractLoader import AbstractLoader
from utility import utils
import numpy as np
import json


class NewAbstracter(AbstractLoader):
    def __init__(self, root, num_neg):
        super(NewAbstracter, self).__init__(root, num_neg)
        self.all_items_set = set(range(self.num_items))

    def load_meta(self):
        with open(os.path.join(self.root, "meta.json")) as f:
            meta = json.loads(f.read())
        self.num_users, self.num_items = meta['num_users'], meta['num_items']

    def get_neg_samples(self, pos_sample):
        # pos_sample: [user, item]
        return list(self.all_items_set.difference({pos_sample[1]}))

    def get_train_data(self):
        fname = os.path.join(self.root, 'train.dat')
        train_data = utils.load_rating_file_as_matrix(fname, splitter=',')
        self.popularity = self.get_popularity(train_data)
        return train_data

    def get_test_data(self):
        fname = os.path.join(self.root, 'test.dat')
        test_pos = utils.load_rating_file_as_list(fname, splitter=',')
        # test_neg = (self.get_neg_samples(pos) for pos in test_pos)
        fname = os.path.join(self.root, 'test.negative.dat')
        test_neg = utils.load_negative_file(fname, splitter=',')
        return test_pos, test_neg

    def get_val_data(self):
        fname = os.path.join(self.root, 'val.dat')
        val_pos = utils.load_rating_file_as_list(fname, splitter=',')
        fname = os.path.join(self.root, 'val.negative.dat')
        val_neg = utils.load_negative_file(fname, splitter=',')
        return val_pos, val_neg

    @staticmethod
    def get_popularity(train_data):
        sum_iteractions = np.array(np.sum(train_data, axis=0)).reshape(-1)
        arg_sort = np.argsort(-sum_iteractions)

        popularity = np.ones(shape=sum_iteractions.shape) * len(sum_iteractions)

        for rank, arg in enumerate(arg_sort):
            popularity[arg] = rank

        return popularity

    def get_sparsity(self):
        return "{0:.3f}%".format(self.sparsity * 100)
