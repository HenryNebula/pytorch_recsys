from numba import jit, prange
from data_pipeline.Evaluator import Evaluator
import numpy as np
import torch.utils.data
from scipy.sparse import coo_matrix
from random import randint


class AbstractLoader(torch.utils.data.Dataset):
    def __init__(self, root, num_neg, implicit=False):
        self.root = root
        self.num_neg = num_neg
        self.implicit = implicit

        self.train_ratings: coo_matrix
        self.test_ratings: coo_matrix
        self.train_ratings, self.test_ratings = self.get_ratings()

        self.num_users, self.num_items = self.train_ratings.shape

        train_rating_csr = self.train_ratings.tocsr()
        self.train_rel_items = [train_rating_csr[idx].indices
                                for idx in range(self.num_users)]

        self.sparse_data_list = list(zip(self.train_ratings.row,
                                         self.train_ratings.col,
                                         self.train_ratings.data))

        self.test_candidates = self.get_test_candidates()

        print("-----------------------------------")
        print("Dataset {} loaded.".format(self.get_dataset_name()))
        print("num_users: {0}, num_items: {1}".format(self.num_users, self.num_items))
        print("-----------------------------------")

    def get_dataset_name(self):
        return "Abstract Loader"

    def get_ratings(self):
        """
        - Return
        -- train_data	csr matrix shaped (num_users, num_items)
        -- test_data	csr matrix shaped (num_users, num_items)
        """
        raise NotImplementedError

    def get_test_candidates(self):
        """
        - Return
        -- test_pos	A list of positive samples (u, i)
        -- test_neg	A list of negative sample list (i1, i2, ....., in) corresponding to u.
        """
        raise NotImplementedError

    def get_evaluator(self):
        return Evaluator(self.test_ratings, self.test_candidates)

    def count(self):
        return self.num_users, self.num_items

    def get_popularity(self):
        sum_iteractions = np.array(np.sum(self.train_ratings, axis=0)).reshape(-1)
        arg_sort = np.argsort(-sum_iteractions)

        popularity = np.ones(shape=sum_iteractions.shape) * len(sum_iteractions)

        for rank, arg in enumerate(arg_sort):
            popularity[arg] = rank

        return popularity

    @staticmethod
    def explicit_to_implicit(rating_mtx: coo_matrix):
        rating_mtx.data = np.ones(shape=(rating_mtx.nnz, ), dtype=np.float)

    @staticmethod
    @jit(nopython=True)
    def find_neg(user_rel_items, item_range, neg_num):
        items = []
        user_rel = set(user_rel_items)
        for t in range(neg_num):
            item = randint(0, item_range - 1)
            while item in user_rel or item in items:
                item = randint(0, item_range - 1)
            items.append(item)
        return items

    def __len__(self):
        return self.train_ratings.nnz

    def __getitem__(self, index):
        # positive
        user, item, label = self.sparse_data_list[index]

        users = np.ones((self.num_neg + 1, ), dtype=np.int32)
        labels = np.zeros((self.num_neg + 1, ), dtype=np.float)
        labels[0] = label

        if self.num_neg > 0:
            neg_items = self.find_neg(self.train_rel_items[int(user)], self.num_items, self.num_neg)

        else:
            neg_items = []

        neg_items.insert(0, item)

        items = np.array(neg_items, dtype=np.int32)

        return users, items, labels

