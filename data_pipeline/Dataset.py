import os, sys
from data_pipeline.AbstractLoader import AbstractLoader
from utility import utils
import pandas as pd


class NewAbstracter(AbstractLoader):
    def __init__(self, root, num_neg):
        super(NewAbstracter, self).__init__(root, num_neg)
        self.all_items_set = set(range(self.num_items))

    def get_neg_samples(self, pos_sample):
        # pos_sample: [user, item]
        return list(self.all_items_set.difference({pos_sample[1]}))

    def get_train_data(self):
        fname = os.path.join(self.root, 'train.dat')
        train_data = utils.load_rating_file_as_matrix(fname, splitter=',')
        self.popularity = self.get_popularity()
        return train_data

    def get_test_data(self):
        fname = os.path.join(self.root, 'test.dat')
        test_pos = utils.load_rating_file_as_list(fname, splitter=',')
        # test_neg = (self.get_neg_samples(pos) for pos in test_pos)
        fname = os.path.join(self.root, 'test.negative.dat')
        test_neg = utils.load_negative_file(fname, splitter=',')
        return test_pos, test_neg
