from graph.BaseMF import BaseMF


class CML(BaseMF):
    '''
    -CML:  Collaborative Metric Learning.
    '''
    def __init__(self, dict_config):
        super(CML, self).__init__(dict_config, embed=True, activate=False, use_dist=True)

    def get_default_loss_type(self):
        return 'MG'

