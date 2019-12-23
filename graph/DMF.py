import torch
from graph.BaseMF import BaseMF

class DMF(BaseMF):
    '''
    -DMF:  abstract class. MF followed by a special function over the l2 distance.
    '''
    def __init__(self, dict_config, norm=2):
        super(DMF, self).__init__(dict_config, embed=True, activate=False)
        self.norm = norm

    def dist2sim(self, dist):
        '''
        A function mapping distance into similarity
        - Input
        -- dist:	torch tensor [N, 1]
        - Return
        -- sim:		torch tensor [N, 1]
        '''
        raise NotImplementedError

    def get_similarity(self, input):
        f_user, f_item = input[0], input[1]

        dist = torch.norm(f_user - f_item, p=self.norm, dim=1)
        dist = dist.view(dist.shape[0], 1)

        sim = self.dist2sim(dist)
        return sim

