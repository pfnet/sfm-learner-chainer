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
    # OK
    poses = pose_vec2mat(poses, xp)

    # TODO
    pixel_coords = utils.generate_2dmeshgrid(H, W, xp)
    cam_coords = pixel2cam(depthes, pixel_coords, K, im_shape, xp=xp)

    # OK
    filler = xp.tile(xp.asarray([0.0, 0.0, 0.0, 1.0], 'f').reshape(1, 1, 4),
                     [N, 1, 1])
    K_ = F.concat([F.concat([K, xp.zeros([N, 3, 1], 'f')], axis=2), filler], axis=1)
    proj_tgt_cam_to_src_pixel = F.matmul(K_, poses)

    # TODO
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel, im_shape)
    """
    hoge = utils.generate_2dmeshgrid(H, W, xp)
    cam_coords2 = pixel2cam2(depthes, hoge, K, im_shape, xp=xp)
    filler = xp.tile(xp.asarray([0.0, 0.0, 0.0, 1.0], 'f').reshape(1, 1, 4),
                     [N, 1, 1])
    K_ = F.concat([F.concat([K, xp.zeros([N, 3, 1], 'f')], axis=2), filler], axis=1)
    proj_tgt_cam_to_src_pixel = F.matmul(K_, poses)
    src_pixel_coords2 = cam2pixel2(cam_coords2, proj_tgt_cam_to_src_pixel, im_shape)
    """
    # print((src_pixel_coords.data == src_pixel_coords2.data).all())
    # print(src_pixel_coords.data[0, :, :2, :2])
    # print(src_pixel_coords2.data[0, :, :2, :2])
    transformed_img = F.spatial_transformer_sampler(imgs, src_pixel_coords)
    return transformed_img
