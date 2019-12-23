import torch
from torch import nn
from torch.nn import functional as F
import os, sys
nowpath = os.getcwd()
sys.path.append(os.path.join(nowpath, '..'))
from utility.loss import l2_loss, bce_loss, bpr_loss, margin_loss
import numpy as np

class BaseMF(nn.Module):
    def __init__(self, dict_config, embed=True, activate=True, use_dist=False):
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
        self.fuse = self.config['fuse']
        self.device = self.config['device']
        self.device = torch.device('cuda:0') if self.device != -1 else torch.device('cpu')

        # specifically for metric learning methods
        self.use_dist = use_dist
        self.square_dist = self.config['square_dist']

        self.embed = embed
        self.activate = activate

        # fake embedding for distributed scenario
        self.fake_user_embedding = nn.Embedding(self.num_users, self.num_factors)
        self.use_fake = False

        self.user_embedding = nn.Embedding(self.num_users, self.num_factors)
        self.item_embedding = nn.Embedding(self.num_items, self.num_factors)

        self.use_user_bias = self.config['use_user_bias']
        if self.use_user_bias:
            self.user_bias = nn.Embedding(self.num_users, 1)

        self.use_item_bias = self.config['use_item_bias']
        if self.use_item_bias:
            self.item_bias = nn.Embedding(self.num_items, 1)

        self.activation = nn.Sigmoid()
        self.flag_index = 1

    def train_with_option(self, fake: bool):
        # fake: True if you want to train item embeddings
        self.use_fake = fake
        self.item_embedding.weight.requires_grad = self.use_fake

    def get_similarity(self, input):
        raise NotImplementedError

    def get_distance(self, input):
        f_user, f_item = input[0], input[1]
        dist = torch.norm(f_user - f_item, p=2, dim=1)
        return dist

    def forward_distance(self, input):
        users, items = input[0], input[1]
        if self.fuse:
            f_user = self.user_embedding(users) * (1 - self.fuse) + self.fake_user_embedding * self.fuse \
                if not self.use_fake else self.fake_user_embedding(users)
        else:
            f_user = self.user_embedding(users) if not self.use_fake else self.fake_user_embedding(users)

        f_item = self.item_embedding(items)

        if self.norm_user:
            f_user = F.normalize(f_user, p=2, dim=1)
        if self.norm_item:
            f_item = F.normalize(f_item, p=2, dim=1)
        dist = self.get_distance([f_user, f_item])
        if self.square_dist:
            dist = dist * dist
        return dist

    def forward_similarity(self, input):
        users, items = input[0], input[1]
        if self.embed:
            if self.fuse:
                f_user = self.user_embedding(users) * (1 - self.fuse) + self.fake_user_embedding(users) * self.fuse \
                    if not self.use_fake else self.fake_user_embedding(users)
            else:
                f_user = self.user_embedding(users) if not self.use_fake else self.fake_user_embedding(users)

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
        if not self.use_dist:
            res = self.forward_similarity(input)
        else:
            res = self.forward_distance(input)
        return res

    def get_predictions(self, input):
        pred = self.forward(input)
        if self.use_dist:
            pred = - pred
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

    def get_sim_loss(self, sim, labels, weights, target=1.0, bpr_margin=0.0, mg_margin=0.5):
        if self.loss_type == 'L2':
            targets = np.array([target, 1 - target]) * 2 - 1
            loss = l2_loss(sim, labels, targets=targets, weights=weights)
        elif self.loss_type == 'BCE':
            if not self.activate:
                sim = self.activation(sim)
            loss = bce_loss(sim, labels, weights=weights, target=target)
        elif self.loss_type == 'BPR':
            loss = bpr_loss(sim, labels, weights=weights, p_index=self.p_index, n_index=self.n_index, bpr_margin=bpr_margin)
        elif self.loss_type == 'MG':
            loss = margin_loss(sim, labels, weights=weights, p_index=self.p_index, n_index=self.n_index, margin=mg_margin)
        else:
            raise(Exception('Error! Invalid loss type {0}.'.format(self.loss_type)))
        return loss

    def get_dist_loss(self, dist, labels, weights, target=1.0, bpr_margin=0.0, mg_margin=0.5):
        if self.loss_type == 'L2':
            sim = - dist
            targets = np.array([1 - target, target])
            loss = l2_loss(sim, labels, weights=weights, targets=targets)
        elif self.loss_type == 'BPR':
            sim = - dist
            loss = bpr_loss(sim, labels, weights, self.p_index, self.n_index, bpr_margin=bpr_margin)
        elif self.loss_type == 'MG':
            sim = - dist
            loss = margin_loss(sim, labels, weights, self.p_index, self.n_index, margin=mg_margin)
        else:
            raise(Exception('Error! Invalid loss type {0}.'.format(self.loss_type)))
        return loss

    def get_loss(self, sim, labels, weights, target=1.0, bpr_margin=0.0, mg_margin=0.5):
        if self.flag_index:
            self.get_index(labels)
            self.flag_index = 0
        if self.loss_type == 'default':
            self.loss_type = self.get_default_loss_type()
        if target != 1.0:
            if target <= 0.5 or target > 1.0:
                raise(Exception('Error! Invalid soft target {0}. The argument target should be within the range (0.5, 1.0).'.format(target)))

        if not self.use_dist:
            loss = self.get_sim_loss(sim, labels, weights, target=target, bpr_margin=bpr_margin, mg_margin=mg_margin)
        else:
            loss = self.get_dist_loss(sim, labels, weights, target=target, bpr_margin=bpr_margin, mg_margin=mg_margin)
        return loss

    def load_model(self, model_data):
        self.load_state_dict(model_data)

    def load_model_from_file(self, path, optimizer='default'):
        print('Loading model from {0}.'.format(path))
        if not os.path.isfile(path):
            raise(Exception('Error! The model file \'{0}\' not found.'.format(path)))
        data = torch.load(path)

        self.load_model(data['model'])
        if optimizer == 'default':
            return None 
        else:
            optimizer.load_state_dict(data['optim'])
            return optimizer

    def load_embedding(self, model_data):
        embedding_keys = ['user_embedding.weight', 'item_embedding.weight', 'fake_user_embedding.weight']
        dict_ = {k: v for k, v in model_data.items() if k in embedding_keys}

        model_dict = self.state_dict()
        print(list(model_dict.keys()))
        model_dict.update(dict_)
        self.load_state_dict(model_dict)

    def load_embedding_from_file(self, path):
        print('Loading embedding from {0}.'.format(path))
        if not os.path.isfile(path):
            raise(Exception('Error! The model file \'{0}\' not fould.'.format(path)))
        data = torch.load(path)
        self.load_embedding(data['model'])
        return self

    ## Specific functions for NeuMF and DoubleGMF
    def load_pretrained_model(self, path):
        raise(Exception('Invalid functions. This function should be used within either NeuMF or DoubleGMF.'))
    def load_pretrained_embedding(self, path):
        raise(Exception('Invalid functions. This function should be used within either NeuMF or DoubleGMF.'))
    def fix_left(self, path):
        raise(Exception('Invalid functions. This function should be used within either NeuMF or DoubleGMF.'))
    def fix_right(self, path):
        raise(Exception('Invalid functions. This function should be used within either NeuMF or DoubleGMF.'))


