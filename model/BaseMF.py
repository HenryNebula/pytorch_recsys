import torch
from torch import nn
from torch.nn import functional as F
import os, sys
nowpath = os.getcwd()
sys.path.append(os.path.join(nowpath, '..'))
from utility.loss import l2_loss, bce_loss, bpr_loss, margin_loss
import numpy as np


class BaseMF(nn.Module):
    def __init__(self, dict_config, embed=True, activate=False):
        super(BaseMF, self).__init__()
        self.config = dict_config
        self.num_users = self.config['num_users']
        self.num_items = self.config['num_items']
        self.num_factors = self.config['num_factors']
        self.loss_type = self.config['loss_type']
        self.norm_user = self.config['norm_user']
        self.norm_item = self.config['norm_item']
        self.multiplier = self.config['multiplier']
        self.bias = self.config['bias']
        self.device = self.config['device']

        # specifically for metric learning methods
        self.square_dist = self.config['square_dist']

        self.embed = embed
        self.activate = activate

        self.user_embedding = nn.Embedding(self.num_users, self.num_factors).double()
        self.item_embedding = nn.Embedding(self.num_items, self.num_factors).double()

        self.use_user_bias = self.config['use_user_bias']
        if self.use_user_bias:
            self.user_bias = nn.Embedding(self.num_users, 1).double()

        self.use_item_bias = self.config['use_item_bias']
        if self.use_item_bias:
            self.item_bias = nn.Embedding(self.num_items, 1).double()

        self.activation = nn.Sigmoid()
        self.flag_index = 1

    def get_similarity(self, input):
        raise NotImplementedError

    def forward_similarity(self, input):
        users, items = input[0], input[1]
        if self.embed:
            f_user = self.user_embedding(users)
            f_item = self.item_embedding(items)

            if self.norm_user:
                f_user = F.normalize(f_user, p=2, dim=1)
            if self.norm_item:
                f_item = F.normalize(f_item, p=2, dim=1)
            sim = self.get_similarity([f_user, f_item])

        else:
            sim = self.get_similarity([users, items])

        if self.use_user_bias:
            user_bias = self.user_bias(users)
            sim = sim + user_bias
        if self.use_item_bias:
            item_bias = self.item_bias(items)
            sim = sim + item_bias

        sim = sim * self.multiplier + self.bias
        if self.activate:
            sim = self.activation(sim)
        return sim

    def forward(self, input):
        res = self.forward_similarity(input)
        return res

    def get_predictions(self, input):
        pred = self.forward(input)
        return pred

    def get_index(self, labels):
        num = labels.shape[0]
        p_num = int(labels.sum(0).detach().cpu().numpy())
        num_neg = int(num / p_num) - 1
    
        p_index_array = np.arange(p_num) * (num_neg + 1)
        self.p_index = torch.Tensor(p_index_array).to(self.device).long()
        n_index_array = [[p_index_array[k1] + 1 + k2 for k2 in range(num_neg)] for k1 in range(p_num)]
        n_index_array = np.reshape(n_index_array, [-1])
        self.n_index = torch.Tensor(n_index_array).to(self.device).long()

    def get_default_loss_type(self):
        return 'BCE'

    def get_sim_loss(self, sim, labels, target=1.0, bpr_margin=0.0, mg_margin=0.5):
        if self.loss_type == 'L2':
            loss = l2_loss(sim, labels)
        elif self.loss_type == 'BCE':
            if not self.activate:
                sim = self.activation(sim)
            loss = bce_loss(sim, labels, target=target)
        elif self.loss_type == 'BPR':
            loss = bpr_loss(sim, labels, p_index=self.p_index, n_index=self.n_index, bpr_margin=bpr_margin)
        elif self.loss_type == 'MG':
            loss = margin_loss(sim, labels, p_index=self.p_index, n_index=self.n_index, margin=mg_margin)
        else:
            raise(Exception('Error! Invalid loss type {0}.'.format(self.loss_type)))
        return loss

    def get_loss(self, sim, labels, target=1.0, bpr_margin=0.0, mg_margin=0.5):
        if self.flag_index:
            self.get_index(labels)
            self.flag_index = 0
        if self.loss_type == 'default':
            self.loss_type = self.get_default_loss_type()
        if target != 1.0:
            if target <= 0.5 or target > 1.0:
                raise(Exception('Error! Invalid soft target {0}. The argument target should be within the range (0.5, 1.0).'.format(target)))

        loss = self.get_sim_loss(sim, labels, target, bpr_margin=bpr_margin, mg_margin=mg_margin)

        return loss


