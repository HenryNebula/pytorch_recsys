import torch
from torch import nn
from model.BaseMF import BaseMF


class LadderGMF(BaseMF):
    def __init__(self, model_config, num_layers=3):
        super(LadderGMF, self).__init__(model_config)

        self.num_layers = num_layers
        self.user_ladder = nn.ModuleList()
        self.item_ladder = nn.ModuleList()
        for j in range(self.num_layers):
            self.user_ladder.append(nn.Sequential(
                nn.Linear(self.num_factors, self.num_factors), 
                nn.ReLU(inplace=True),
                nn.Linear(self.num_factors, self.num_factors)))
            self.item_ladder.append(nn.Sequential(
                nn.Linear(self.num_factors, self.num_factors), 
                nn.ReLU(inplace=True),
                nn.Linear(self.num_factors, self.num_factors)))

        self.mapping = nn.Linear(self.num_layers + 1, 1)

    def get_similarity(self, input):
        f_user, f_item = input[0], input[1]
        
        N = f_user.shape[0]
        inter_features = []
        inter_features.append((f_user * f_item).sum(1).view(N,1))
        for j in range(self.num_layers):
            user_layer = self.user_ladder[j]
            f_user = user_layer(f_user)
            item_layer = self.item_ladder[j]
            f_item = item_layer(f_item)
            inter_features.append((f_user * f_item).sum(1).view(N,1))

        inter_features = torch.cat(inter_features, dim=1)
        sim = self.mapping(inter_features)
        return sim

