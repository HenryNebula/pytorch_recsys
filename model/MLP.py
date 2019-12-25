import torch
from graph.BaseMF import BaseMF
from utility import model_utils


class MLP(BaseMF):
    def __init__(self, dict_config):
        super(MLP, self).__init__(dict_config, embed=True, activate=True)

        self.mapping = model_utils.make_fc_layers([32, 16, 8, 1], in_channels=2*self.num_factors)

    def get_similarity(self, input):
        f_user, f_item = input[0], input[1]

        features = torch.cat((f_user, f_item), dim=1)
        sim = self.mapping(features)
        return sim

