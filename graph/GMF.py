from torch import nn
from graph.BaseMF import BaseMF


class GMF(BaseMF):
    def __init__(self, dict_config):
        super(GMF, self).__init__(dict_config, embed=True, activate=True)

        self.mapping_flag = dict_config['mapping']
        self.mapping = nn.Linear(1,1)

    def get_similarity(self, input):
        f_user, f_item = input[0], input[1]

        dot = (f_user * f_item).sum(1)
        dot = dot.view(dot.shape[0], 1)
        if self.mapping_flag:
            sim = self.mapping(dot)
        else:        
            sim = dot
        return sim

