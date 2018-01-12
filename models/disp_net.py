# Original implementation with TF : https://github.com/tinghuiz/SfMLearner/blob/master/nets.py
import chainer
from chainer import functions as F
from chainer import links as L

# Range of disparity/inverse depth values
DISP_SCALING = 10
MIN_DISP = 0.01


def resize_like(inputs, ref):
    if inputs.shape[2] == ref.shape[2] and inputs.shape[3] == ref.shape[3]:
        return inputs
    return F.resize_images(inputs, ref.shape[2:])


class DispNet(chainer.Chain):
    def __init__(self, activation=F.relu):
        super(DispNet, self).__init__()
        self.activation = activation
        with self.init_scope():
            # Deep...
            self.c1 = L.Convolution2D(None, 32, ksize=8, stride=2, pad=3)
            self.c1b = L.Convolution2D(None, 32, ksize=7, stride=1, pad=3)
            self.c2 = L.Convolution2D(None, 64, ksize=6, stride=2, pad=2)
            self.c2b = L.Convolution2D(None, 64, ksize=5, stride=1, pad=2)
            self.c3 = L.Convolution2D(None, 128, ksize=4, stride=2, pad=1)
            self.c3b = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
            self.c4 = L.Convolution2D(None, 256, ksize=4, stride=2, pad=1)
            self.c4b = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
            self.c5 = L.Convolution2D(None, 512, ksize=4, stride=2, pad=1)
            self.c5b = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)
            self.c6 = L.Convolution2D(None, 512, ksize=4, stride=2, pad=1)
            self.c6b = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)
            self.c7 = L.Convolution2D(None, 512, ksize=4, stride=2, pad=1)
            self.c7b = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)

            self.dc7 = L.Deconvolution2D(None, 512, ksize=4, stride=2, pad=1)
            self.idc7 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)
            self.dc6 = L.Deconvolution2D(None, 512, ksize=4, stride=2, pad=1)
            self.idc6 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)
            self.dc5 = L.Deconvolution2D(None, 256, ksize=4, stride=2, pad=1)
            self.idc5 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
            self.dc4 = L.Deconvolution2D(None, 128, ksize=4, stride=2, pad=1)
            self.idc4 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
            self.dispout4 = L.Convolution2D(None, 1, ksize=3, stride=1, pad=1)
            self.dc3 = L.Deconvolution2D(None, 64, ksize=4, stride=2, pad=1)
            self.idc3 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
            self.dispout3 = L.Convolution2D(None, 1, ksize=3, stride=1, pad=1)
            self.dc2 = L.Deconvolution2D(None, 32, ksize=4, stride=2, pad=1)
            self.idc2 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1)
            self.dispout2 = L.Convolution2D(None, 1, ksize=3, stride=1, pad=1)
            self.dc1 = L.Deconvolution2D(None, 16, ksize=4, stride=2, pad=1)
            self.idc1 = L.Convolution2D(None, 16, ksize=3, stride=1, pad=1)
            self.dispout1 = L.Convolution2D(None, 1, ksize=3, stride=1, pad=1)

    def __call__(self, x_target):
        # x_target: chainer.Variable of shape = [N, 3, H, W]
        # There might be dimension mismatch due to uneven down/up-sampling
        H, W = x_target.shape[2:]
        normalizer = lambda z: z
        h = x_target
        h = self.activation(normalizer(self.c1(h)))
        h = self.activation(normalizer(self.c1b(h)))
        h_c1b = h
        h = self.activation(normalizer(self.c2(h)))
        h = self.activation(normalizer(self.c2b(h)))
        h_c2b = h
        h = self.activation(normalizer(self.c3(h)))
        h = self.activation(normalizer(self.c3b(h)))
        h_c3b = h
        h = self.activation(normalizer(self.c4(h)))
        h = self.activation(normalizer(self.c4b(h)))
        h_c4b = h
        h = self.activation(normalizer(self.c5(h)))
        h = self.activation(normalizer(self.c5b(h)))
        h_c5b = h
        h = self.activation(normalizer(self.c6(h)))
        h = self.activation(normalizer(self.c6b(h)))
        h_c6b = h
        h = self.activation(normalizer(self.c7(h)))
        h = self.activation(normalizer(self.c7b(h)))

        h = self.activation(normalizer(self.dc7(h)))
        # There might be dimension mismatch due to uneven down/up-sampling
        # Resize by bilinear interpolation.
        # (by nearest neighbor sampling in the original implemntation.)
        h = resize_like(h, h_c6b)
        h = F.concat([h, h_c6b], axis=1)
        h = self.activation(normalizer(self.idc7(h)))

        h = self.activation(normalizer(self.dc6(h)))
        h = resize_like(h, h_c5b)
        h = F.concat([h, h_c5b], axis=1)
        h = self.activation(normalizer(self.idc6(h)))

        h = self.activation(normalizer(self.dc5(h)))
        h = resize_like(h, h_c4b)
        h = F.concat([h, h_c4b], axis=1)
        h = self.activation(normalizer(self.idc5(h)))

        h = self.activation(normalizer(self.dc4(h)))
        h = F.concat([h, h_c3b], axis=1)
        h = self.activation(normalizer(self.idc4(h)))
        disp4 = DISP_SCALING * F.sigmoid(self.dispout4(h)) + MIN_DISP
        disp4_up = F.resize_images(disp4, (H // 4, W // 4))

        h = self.activation(normalizer(self.dc3(h)))
        h = F.concat([h, h_c2b, disp4_up], axis=1)
        h = self.activation(normalizer(self.idc3(h)))
        disp3 = DISP_SCALING * F.sigmoid(self.dispout3(h)) + MIN_DISP
        disp3_up = F.resize_images(disp3, (H // 2, W // 2))

        h = self.activation(normalizer(self.dc2(h)))
        h = F.concat([h, h_c1b, disp3_up], axis=1)
        h = self.activation(normalizer(self.idc2(h)))
        disp2 = DISP_SCALING * F.sigmoid(self.dispout2(h)) + MIN_DISP
        disp2_up = F.resize_images(disp2, (H, W))

        h = self.activation(normalizer(self.dc1(h)))
        h = F.concat([h, disp2_up], axis=1)
        h = self.activation(normalizer(self.idc1(h)))
        disp1 = DISP_SCALING * F.sigmoid(self.dispout1(h)) + MIN_DISP

        return [disp1, disp2, disp3, disp4]
