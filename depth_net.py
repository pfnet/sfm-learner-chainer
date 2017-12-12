import chainer
from chainer import functions as F

class DepthNet(chainer.Chain):

    def __init__(self):
        super(DepthNet, self).__init__()
        raise NotImplementedError


    def __call__(self, x):
        # x.shape = [N, C, H, W]
        raise NotImplementedError

