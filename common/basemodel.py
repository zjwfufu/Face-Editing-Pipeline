import torch

# base nodel
class BaseModel(object):
    def __init__(self, config):
        self.network = None
        self.use_cuda = None

    def _load_model(self, network=None, weights=None):
        state_dict = torch.load(weights)
        network.load_state_dict(state_dict)
        network.eval()
        if self.use_cuda:
            network.cuda()
        return network

    # 1. preprocess
    def preprocess(self):
        raise NotImplementedError('>>> preprocess function must be implemented')

    # 2. network process
    def process(self):
        # self.network.infer()
        raise NotImplementedError('>>> process function must be implemented')

    # 3. postprocess
    def postprocess(self):
        raise NotImplementedError('>>> postprocess function must be implemented')
