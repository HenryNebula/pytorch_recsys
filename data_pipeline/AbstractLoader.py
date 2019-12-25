import os, sys
from dataloader.Evaluator import Evaluator
import numpy as np
import torch.utils.data
from scipy.sparse import lil_matrix


class AbstractLoader(torch.utils.data.Dataset):
    def __init__(self, root, num_neg):
        self.root = root
        self.num_neg = num_neg

        self.train_data: lil_matrix = self.get_train_data()
        self.get_data_info()

    def get_train_data(self):
        '''
        - Return
        -- train_data	lil matrix shaped (num_users, num_items)
        '''
        raise NotImplementedError

    def get_test_data(self):
        '''
        - Return
        -- test_pos	A list of positive samples (u, i)
        -- test_neg	A list of negative sample list (i1, i2, ....., in) corresponding to u.
        '''
        raise NotImplementedError

    def get_val_data(self):
        '''
        - Return
        -- val_pos	A list of positive samples (u, i)
        -- val_neg	A list of negative sample list (i1, i2, ....., in) corresponding to u.
        '''
        raise NotImplementedError

    def get_evaluator(self, is_val):
        if is_val:
            pos, neg = self.get_val_data()
        else:
            pos, neg = self.get_test_data()
        return Evaluator(pos, neg)

    def get_data_info(self):
        mat_dok = self.train_data.todok()
        self.train_list = list(mat_dok.keys())
        self.num_users = self.train_data.shape[0]
        self.num_items = self.train_data.shape[1]
        self.sparsity = 1 - self.train_data.nnz / (self.num_items * self.num_users)

    def count(self):
        return self.num_users, self.num_items
 
    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        users, items, labels = [], [], []

        # positive
        user = self.train_list[index][0]
        item = self.train_list[index][1]
        users.append(user)
        items.append(item)
        labels.append(1)

        # negative
        user_ll = self.train_data.rows[user]

        for t in range(self.num_neg):
            item = np.random.randint(self.num_items)
            while item in user_ll:
                item = np.random.randint(self.num_items)
            users.append(user)
            items.append(item)
            labels.append(0)

        users, items, labels = np.array(users), np.array(items), np.array(labels)
        return users, items, labels

