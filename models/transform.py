# Transformation from target view to source
# See https://github.com/tinghuiz/SfMLearner/blob/master/utils.py
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import cuda
import numpy as np
from models import utils


def transform(imgs, depthes, poses, K):
    """
    Args:
        imgs: chainer.Variable of shape [N, 3, H, W]
        depthes: chainer.Variable of shape [N, 1, H, W]
        poses: chainer.Variable of shape [N, 6]
        K: [N, 3, 3]
    Return:
        transformed images of shape [N, 3, H, W]
    """
    xp = cuda.get_array_module(imgs.data)
    print(imgs.shape, depthes.shape, poses.shape, K.shape)
    N, _, H, W = imgs.shape
    # Generate target coordinate
    p_t = utils.generate_2dmeshgrid(H, W, xp)
    # Convert pixel coordinates to the camera frame
    cam_coords = F.matmul(F.batch_inv(K),
                          xp.broadcast_to(pixel_coords.reshape(1, 3, H*W),
                                          (N, 3, H*W))
                 ).reshape(N, 3, H, W)
    DKinvp = F.broadcast_to(depthes, (N, 3, H, W)) * cam_coords 
    DKinvp = F.concat([DKinvp, xp.ones((N, 1, H, W), xp.float32)], axis=1)  # Convert to homgeneneous vectors
    # Convert pose vector to matrix
    T = utils.pose_vec2mat(poses, xp)
    filter = xp.tile(xp.asarray([0.0, 0.0, 0.0, 1.0], xp.float32).reshape(1, 1, 4), [N, 1, 1])
    _K = F.concat([F.concat([K, xp.zeros([N, 3, 1], xp.float32)], axis=2), filter], axis=1)
    KT = F.batch_matmul(_K, T)
    p_s = F.batch_matmul(KT, F.reshape(DKinvp, (N, 4, -1)))  # Source coordinate
    p_s_xy = p_s[:, 0:2, :]
    p_s_xy /= F.broadcast_to(p_s[:, 2:3, :], p_s.shape) + 1e-10
    p_s_xy = F.reshape(p_s_xy, (N, 2, H, W))
    transformed_img = F.spatial_transformer_sampler(imgs, p_s_xy)
    return transformed_img
