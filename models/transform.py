# Transformation from target view to source
# See https://github.com/tinghuiz/SfMLearner/blob/master/utils.py
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import cuda
import numpy as np
from models import utils
from models.utils import *

def transform(imgs, depthes, poses, K):
    """
    Args:
        imgs(Variable): Source images. Shape is [N, 3, H, W]
        depthes(Variable): Predicted depthes. Shape is [N, 1, H, W]
        poses(Variable): Predicted poses. Shape is [N, 6]
        K: [N, 3, 3]
    Return:
        transformed images of shape [N, 3, H, W]
    """
    xp = cuda.get_array_module(imgs.data)
    im_shape = imgs.shape
    N, _, H, W = im_shape


    pixel_coords = utils.generate_2dmeshgrid(H, W, xp)
    cam_coords = pixel2cam(depthes, pixel_coords, K, im_shape, xp=xp)

    filler = xp.tile(xp.asarray([0.0, 0.0, 0.0, 1.0], 'f').reshape(1, 1, 4),
                     [N, 1, 1])
    K_ = F.concat([F.concat([K, xp.zeros([N, 3, 1], 'f')], axis=2), filler], axis=1)
    poses = pose_vec2mat(poses, filler, xp)
    proj_tgt_cam_to_src_pixel = F.matmul(K_, poses)

    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel, im_shape)
    transformed_img = F.spatial_transformer_sampler(imgs, src_pixel_coords)
    return transformed_img
