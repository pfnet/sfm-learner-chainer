import chainer
from chainer import functions as F


class PoseAndExpNet(chainer.Chain):

    def __init__(self):
        super(PoseAndExpNet, self).__init__()
        raise NotImplementedError

    def __call__(self, x):
        # x.shape = [N, C, H, W]
        raise NotImplementedError

