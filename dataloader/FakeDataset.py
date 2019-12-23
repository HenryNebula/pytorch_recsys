import numpy as np
from dataloader.AbstractLoader import AbstractLoader
from dataloader.NewAbstracter import NewAbstracter
from scipy.sparse import lil_matrix


class FakeDataset(AbstractLoader):
    def __init__(self, real_dataset: NewAbstracter):
        super(FakeDataset, self).__init__(real_dataset.root, real_dataset.num_neg)

        self.filled_data = {}
        self.full_train_list = []
        self.real_dataset = real_dataset
        self.thresh = 1 - (2 * (1 - self.real_dataset.sparsity)) # sparsity thresh for pos_sampling

    def update_data(self, new_data):
        self.train_data = new_data['train_data']
        self.get_data_info()
        mat_dok = self.train_data.todok()
        self.full_train_list = list(mat_dok.keys())

    def get_train_data(self):
        return lil_matrix(np.random.rand(5,5))

    def get_test_data(self):
        return self.real_dataset.get_test_data

    def get_val_data(self):
        return self.real_dataset.get_val_data

    def refresh_trainlist(self):
        if self.sparsity < self.thresh:
            thresh_pos_items = int((1 - self.thresh) * self.num_items * self.num_users)
            chosen_idx = np.random.choice(range(len(self.full_train_list)), thresh_pos_items, replace=False)
            self.train_list = [self.full_train_list[idx] for idx in chosen_idx]
            print("Apply positive down-sampling, new sparsity {0:.2f}".format(len(self.train_list)/(self.num_items * self.num_users)))
        else:
            print("No need for down-sampling")


    # def __getitem__(self, index):
    #     users, items, labels = [], [], []
    #
    #     # positive
    #     user = self.train_list[index][0]
    #     item = self.train_list[index][1]
    #     users.append(user)
    #     items.append(item)
    #     labels.append(1)
    #
    #     user_ll = self.train_data.rows[user]
    #     if len(user_ll) >= (1 - self.thresh) * self.num_items:
    #         # filter out positive samples first
    #         pos_set = set(user_ll)
    #
    #         # use set for neg_sampling
    #         neg_set = set(range(self.num_items)).difference(pos_set)
    #
    #         if len(neg_set) < self.num_neg:
    #             if len(neg_set) == 0:
    #                 assert False, "All-One Row, Error!"
    #             else:
    #                 residue_num = self.num_neg - len(neg_set)
    #                 residue = np.random.choice(list(neg_set), size=residue_num, replace=True).tolist()
    #                 neg_item = list(neg_set) + residue
    #         else:
    #             neg_item = np.random.choice(list(neg_set), size=self.num_neg, replace=False).tolist()
    #
    #         users.extend([user]*self.num_neg)
    #         items.extend(neg_item)
    #         labels.extend([0]*self.num_neg)
    #     else:
    #         for t in range(self.num_neg):
    #             item = np.random.randint(self.num_items)
    #             while item in user_ll:
    #                 item = np.random.randint(self.num_items)
    #             users.append(user)
    #             items.append(item)
    #             labels.append(0)
    #     users, items, labels = np.array(users), np.array(items), np.array(labels)
    #     return users, items, labels

