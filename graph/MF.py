from graph.BaseMF import BaseMF


class MF(BaseMF):
    def __init__(self, dict_config):
        super(MF, self).__init__(dict_config, embed=True, activate=False)

    def get_default_loss_type(self):
        return 'L2'

    def get_similarity(self, input):
        f_user, f_item = input[0], input[1]

        dot = (f_user * f_item).sum(1)
        dot = dot.view(dot.shape[0], 1)
        sim = dot
        return sim

