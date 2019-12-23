import torch
from torch import nn
from graph.BaseMF import BaseMF

class PMF(BaseMF):
    '''
    -PMF:  MF followed by polynomial mapping.
    '''
    def __init__(self, dict_config, p_order=3):
        super(PMF, self).__init__(dict_config, embed=True, activate=True)
        self.p_order = p_order

        self.mapping = nn.Linear(p_order, 1)

    def get_similarity(self, input):
        f_user, f_item = input[0], input[1]

        dot = (f_user * f_item).sum(1)
        dot = dot.view(dot.shape[0], 1)
        features = [dot ** (k+1) for k in range(self.p_order)]
        features = torch.cat(features, dim=1)

        sim = self.mapping(features)
        return sim

