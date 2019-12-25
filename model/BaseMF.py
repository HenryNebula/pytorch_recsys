import torch
from torch import nn
from torch.nn import functional as F
from model.Loss import l2_loss, bce_loss, bpr_loss, margin_loss
import numpy as np


class BaseMF(nn.Module):
    def __init__(self, model_config):
        super(BaseMF, self).__init__()
        self.config = model_config
        self.num_users = self.config["num_users"]
        self.num_items = self.config["num_items"]
        self.num_factors = self.config["num_factors"]
        self.loss_type = self.config["loss_type"]
        self.norm_user = self.config["norm_user"]
        self.norm_item = self.config["norm_item"]
        self.multiplier = self.config["multiplier"]
        self.bias = self.config["bias"]
        self.device = self.config["device"]

        # specifically for metric learning methods
        self.square_dist = self.config["square_dist"]

        self.implicit = self.config["implicit"]
        self.num_neg = self.config["num_neg"]

        self.user_embedding = nn.Embedding(self.num_users, self.num_factors).double()
        self.item_embedding = nn.Embedding(self.num_items, self.num_factors).double()

        self.use_user_bias = self.config["use_user_bias"]
        if self.use_user_bias:
            self.user_bias = nn.Embedding(self.num_users, 1).double()

        self.use_item_bias = self.config["use_item_bias"]
        if self.use_item_bias:
            self.item_bias = nn.Embedding(self.num_items, 1).double()

        self.p_index = torch.tensor([])
        self.n_index = torch.tensor([])

    def get_similarity(self, input):
        raise NotImplementedError

    def forward_similarity(self, input):
        users, items = input[0], input[1]
        f_user = self.user_embedding(users)
        f_item = self.item_embedding(items)

        if self.norm_user:
            f_user = F.normalize(f_user, p=2, dim=1)
        if self.norm_item:
            f_item = F.normalize(f_item, p=2, dim=1)
        sim = self.get_similarity([f_user, f_item])

        if self.use_user_bias:
            user_bias = self.user_bias(users)
            sim = sim + user_bias
        if self.use_item_bias:
            item_bias = self.item_bias(items)
            sim = sim + item_bias

        sim = sim * self.multiplier + self.bias
        return sim

    def forward(self, input):
        res = self.forward_similarity(input)
        return res

    def get_predictions(self, input):
        pred = self.forward(input)
        return pred

    def reset_index(self, labels):
        # assume that in every batch, positive and negative samples appear in the same relative positions
        self.p_index = torch.nonzero(labels > 0).view(-1)
        self.n_index = torch.nonzero(labels == 0).view(-1)

    def get_default_loss_type(self):
        return "L2"

    def get_sim_loss(self, sim, labels, bpr_margin=0.0, mg_margin=0.5):
        regression_loss = ["L2"]
        ranking_loss = ["BPR", "MG"]

        if (not self.implicit) and (self.loss_type not in regression_loss):
            raise ValueError("{} loss can not be used when dealing with explicit ratings; "
                             "please set implicit to true first")

        if self.loss_type in ranking_loss and self.num_neg <= 0:
            raise ValueError("Ranking loss {} only works along with negative sampling; "
                             "please increase num_neg".format(self.loss_type))

        if self.loss_type in ranking_loss and (self.p_index.nelement() == 0 or self.n_index.nelement() == 0):
            self.reset_index(labels)

        if self.loss_type == "L2":
            loss = l2_loss(sim, labels)
        elif self.loss_type == "BCE":
            loss = bce_loss(sim, labels)
        elif self.loss_type == "BPR":
            loss = bpr_loss(sim, labels, p_index=self.p_index, n_index=self.n_index, bpr_margin=bpr_margin)
        elif self.loss_type == "MG":
            loss = margin_loss(sim, labels, p_index=self.p_index, n_index=self.n_index, margin=mg_margin)
        else:
            raise(ValueError("Invalid loss type {0}.".format(self.loss_type)))
        return loss

    def get_loss(self, sim, labels, bpr_margin=0.0, mg_margin=0.5):
        if self.loss_type == "default":
            self.loss_type = self.get_default_loss_type()

        loss = self.get_sim_loss(sim, labels, bpr_margin=bpr_margin, mg_margin=mg_margin)

        return loss


