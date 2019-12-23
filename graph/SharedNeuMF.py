import torch
from torch import nn
from graph.BaseMF import BaseMF
import utility.utils as utils

class SharedNeuMF(BaseMF):
    def __init__(self, dict_config):
        super(SharedNeuMF, self).__init__(dict_config, embed=True, activate=True)
        self.mapping_flag = dict_config['mapping'] 

        self.GMF_mapping = nn.Linear(1,1)
        self.MLP_mapping = utils.make_fc_layers([32, 16, 8, 1], in_channels=2*self.num_factors)
        self.mapping = nn.Linear(2,1)

    def get_similarity(self, input):
        f_user, f_item = input[0], input[1]

        # GMF
        dot = (f_user * f_item).sum(1)
        dot = dot.view(dot.shape[0], 1)
        if self.mapping_flag:
            sim = self.GMF_mapping(dot)
        else:        
            sim = dot
        sim_GMF = self.activation(sim)

        # MLP
        features = torch.cat((f_user, f_item), dim=1)
        features = self.MLP_mapping(features)
        sim_MLP = self.activation(features)

        # ensemble
        features = torch.cat((sim_GMF, sim_MLP), dim=1)
        sim = self.mapping(features)
        return sim

