from graph.DMF import DMF
import torch


class DMF_A(DMF):
    '''
    -DMF_A:  DMF using function f(x)=1-x^2/2.
    '''
    def __init__(self, dict_config, norm=2):
        super(DMF_A, self).__init__(dict_config, norm=norm)

    def dist2sim(self, dist):
        sim = 1 - dist * dist / 2
        return sim


class DMF_C(DMF):
    '''
    -DMF_C:  DMF using negative exponential function f(x)=1-x.
    '''

    def __init__(self, dict_config, norm=2):
        super(DMF_C, self).__init__(dict_config, norm=norm)

    def dist2sim(self, dist):
        sim = 1 - dist
        return sim


class DMF_B(DMF):
    '''
    -DMF_B:  DMF using negative exponential function f(x)=exp(-x).
    '''

    def __init__(self, dict_config, norm=2):
        super(DMF_B, self).__init__(dict_config, norm=norm)

    def dist2sim(self, dist):
        sim = torch.exp(-dist)
        return sim


class DMF_D(DMF):
    '''
    -DMF_D:  DMF using negative exponential function f(x)=2*exp(-x)-1.
    '''
    def __init__(self, dict_config, norm=2):
        super(DMF_D, self).__init__(dict_config, norm=norm)

    def dist2sim(self, dist):
        sim = 2 * torch.exp(-dist) - 1
        return sim


