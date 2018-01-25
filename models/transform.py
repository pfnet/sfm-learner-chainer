# Transformation from target view to source
# See https://github.com/tinghuiz/SfMLearner/blob/master/utils.py
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import cuda
import numpy as np
import cupy as cp
from models import utils
from models.utils import *
from spatial_transformer_sampler_interp import spatial_transformer_sampler_interp

def euler2mat(r, xp=np):
    """Converts euler angles to rotation matrix

    Args:
        r: rotation angle(x, y, z). Shape is (N, 3).
    Returns:
        Rotation matrix corresponding to the euler angles. Shape is (N, 3, 3).
    """
    batchsize = r.shape[0]
    # start, stop = create_timer()
    zeros = xp.zeros((batchsize), dtype='f')
    ones = xp.ones((batchsize), dtype='f')
    r = F.clip(r, -np.pi, np.pi)
    cos_r = F.cos(r)
    sin_r = F.sin(r)

    zmat = F.stack([cos_r[:, 2], -sin_r[:, 2], zeros,
                    sin_r[:, 2], cos_r[:, 2], zeros,
                    zeros, zeros, ones], axis=1).reshape(batchsize, 3, 3)

    ymat = F.stack([cos_r[:, 1], zeros, sin_r[:, 1],
                    zeros, ones, zeros,
                    -sin_r[:, 1], zeros, cos_r[:, 1]], axis=1).reshape(batchsize, 3, 3)

    xmat = F.stack([ones, zeros, zeros,
                    zeros, cos_r[:, 0], -sin_r[:, 0],
                    zeros, sin_r[:, 0], cos_r[:, 0]], axis=1).reshape(batchsize, 3, 3)
    # #print_timer(start, stop, 'create matrix')
    rotMat = F.batch_matmul(F.batch_matmul(xmat, ymat), zmat)
    return rotMat


def pose_vec2mat(vec, filler, xp=np):
    """Converts 6DoF parameters to transformation matrix

    Args:
        vec: 6DoF parameters in the order of rx, ry, rz, tx, ty, tz -- [N, 6]
    Returns:
        A transformation matrix -- [N, 4, 4]
    """
    # start, stop = create_timer()
    r, t = vec[:, :3], vec[:, 3:]
    rot_mat = euler2mat(r, xp=xp)
    # print_timer(start, stop, 'euler2mat')
    batch_size = rot_mat.shape[0]
    t = t.reshape(batch_size, 3, 1)
    transform_mat = F.dstack((rot_mat, t))
    transform_mat = F.hstack((transform_mat, filler))
    return transform_mat


filler = None

def proj_tgt_to_src(vec, K, N, xp=np, use_cpu=True):
    """Projection matrix from target to sources.

    Args:
        vec(Variable): Shape is (N, 6).
        K(array): Shape is (N, 3, 3).
        N(int): Batch size.

    Returns:
        Variable: Projection matrix.
    """
    is_transfer = False
    if xp == cp and use_cpu:
        vec = gpu2cpu(vec)
        K = chainer.cuda.to_cpu(K)
        xp = np
        is_transfer = True

    global filler
    if filler is None or filler.shape[0] != N:
        filler = xp.tile(xp.asarray([0.0, 0.0, 0.0, 1.0], 'f').reshape(1, 1, 4),
                         [N, 1, 1])
    K_ = F.concat([F.concat([K, xp.zeros([N, 3, 1], 'f')], axis=2), filler], axis=1)
    poses = pose_vec2mat(vec, filler, xp)
    proj_tgt_cam_to_src_pixel = F.batch_matmul(K_, poses)
    if is_transfer:
        proj_tgt_cam_to_src_pixel = cpu2gpu(proj_tgt_cam_to_src_pixel)
    return proj_tgt_cam_to_src_pixel


def pixel2cam(depthes, pixel_coords, intrinsics, im_shape, xp=np):
    """Converter from pixel coordinates to camera coordinates.

    Args:
        depthes(Variable): Shape is (N, 3, H*W)
        pixel_coords(array): Shape is (N, 3, H*W)
        intrinsics(array): Shape is (N, 3, 3)
    Returns:
        cam_coords(Variable): Shape is (N, 3, H*W)
    """
    N, _, H, W = im_shape
    cam_coords = F.batch_matmul(F.batch_inv(intrinsics),
                                pixel_coords)
    cam_coords = depthes * cam_coords
    cam_coords = F.concat((cam_coords, xp.ones((N, 1, H*W), 'f')), axis=1)
    return cam_coords

def cam2pixel(cam_coords, proj, im_shape):
    """Conveter from camera coordinates to pixel coordinates.

    Args:
        cam_coords(Variable): Shape is (N, 4, H*W).
        proj(Variable): Shape is (N, 4, 4).

    Returns:
        Variable: Pixel coordinates.
    """
    N, _, H, W = im_shape
    unnormalized_pixel_coords = F.batch_matmul(proj, cam_coords)
    z = unnormalized_pixel_coords[:, 2:3, :] + 1e-10
    p_s_x = (unnormalized_pixel_coords[:, 0:1] / z) / ((W - 1) / 2.) - 1
    p_s_y = (unnormalized_pixel_coords[:, 1:2] / z) / ((H - 1) / 2.) - 1
    p_s_xy = F.concat((p_s_x, p_s_y), axis=1)
    p_s_xy = F.reshape(p_s_xy, (N, 2, H, W))
    return p_s_xy

meshgrid = None

def generate_2dmeshgrid(H, W, N, xp=np):
    """Generate 2d meshgrid.

    Returns:
        Array: Shape is (N, 3, H*W)
    """
    global meshgrid
    if meshgrid is None or meshgrid.shape[2] != H*W:
        ys, xs = xp.meshgrid(
            xp.arange(0, H, dtype=np.float32),
            xp.arange(0, W, dtype=np.float32), indexing='ij',
            copy=False
        )
        meshgrid = xp.concatenate(
            [xs[None], ys[None], xp.ones((1, H, W), dtype=np.float32)],
            axis=0).reshape(1, 3, H*W)
        meshgrid = F.broadcast_to(meshgrid, (N, 3, H*W))
    return meshgrid

def projective_inverse_warp(imgs, depthes, poses, K):
    """
    Args:
        imgs(Variable): Source images. Shape is [N, 3, H, W]
        depthes(Variable): Predicted depthes. Shape is [N, 3, H*W]
        poses(Variable): Predicted poses. Shape is [N, 6]
        K(array): [N, 3, 3]
    Return:
        transformed images of shape [N, 3, H, W]
    """
    xp = cuda.get_array_module(imgs.data)
    im_shape = imgs.shape
    N, _, H, W = im_shape

    # start, stop = create_timer()
    proj_tgt_cam_to_src_pixel = proj_tgt_to_src(poses, K, N, xp=xp,
                                                use_cpu=True)
    # print_timer(start, stop, 'proj')

    # start, stop = create_timer()
    pixel_coords = generate_2dmeshgrid(H, W, N, xp)
    # print_timer(start, stop, 'mesh grid')

    # start, stop = create_timer()
    cam_coords = pixel2cam(depthes, pixel_coords, K, im_shape, xp=xp)
    # print_timer(start, stop, 'pixel2cam')

    # start, stop = create_timer()
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel, im_shape)
    # print_timer(start, stop, 'cam2pixel')

    # start, stop = create_timer()
    # transformed_img = F.spatial_transformer_sampler(imgs, src_pixel_coords)
    transformed_img = spatial_transformer_sampler_interp(imgs, src_pixel_coords)
    # print_timer(start, stop, 'spatial transformer')
    return transformed_img
