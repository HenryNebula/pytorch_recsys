import torch
from torch import nn
import os, sys
from graph.BaseMF import BaseMF
from graph.GMF import GMF
from graph.MLP import MLP
from torch import optim


class NeuMF(BaseMF):
    def __init__(self, dict_config):
        super(NeuMF, self).__init__(dict_config, embed=False, activate=True)

        self.GMF = GMF(dict_config)
        self.MLP = MLP(dict_config)
        self.mapping = nn.Linear(2,1)

    def fix_left(self, optimizer):
        for param in self.GMF.parameters():
            param.requires_grad = False
        groups = optimizer.param_groups
        lr_, betas_ = groups[0]['lr'], groups[0]['betas']
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr = lr_, betas = betas_)
        return optimizer

    def fix_right(self, optimizer):
        for param in self.MLP.parameters():
            param.requires_grad = False
        groups = optimizer.param_groups
        lr_, betas_ = groups[0]['lr'], groups[0]['betas']
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr = lr_, betas = betas_)
        return optimizer

    def load_pretrained_embedding(self, model_data):
        model_data_1, model_data_2 = model_data[0], model_data[1]
        if model_data_1 != 'default':
            print("Loading GMF embedding in NeuMF...")
            self.GMF.load_embedding_from_file(model_data_1)
        if model_data_2 != 'default':
            print("Loading MLP embedding in NeuMF...")
            self.MLP.load_embedding_from_file(model_data_2)

    def load_pretrained_model(self, model_data):
        model_data_1, model_data_2 = model_data[0], model_data[1]
        if model_data_1 != 'default':
            print("Loading GMF model in NeuMF...")
            self.GMF.load_model_from_file(model_data_1)
        if model_data_2 != 'default':
            print("Loading MLP model in NeuMF...")
            self.MLP.load_model_from_file(model_data_2)

    def get_similarity(self, input):
        users, items = input[0], input[1]
        sim_GMF = self.GMF([users, items])
        sim_MLP = self.MLP([users, items])

        features = torch.cat((sim_GMF, sim_MLP), dim=1)
        sim = self.mapping(features)
        return sim

