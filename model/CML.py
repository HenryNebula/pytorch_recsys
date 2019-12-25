from model.BaseMF import BaseMF


class CML(BaseMF):
    """
    -CML:  Collaborative Metric Learning.
    """
    def __init__(self, model_config):
        super(CML, self).__init__(model_config)

    def get_default_loss_type(self):
        return "MG"

