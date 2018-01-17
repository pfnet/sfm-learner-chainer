import numpy as np
from chainer import functions as F


def euler2mat(r, xp=np):
    """Converts euler angles to rotation matrix

       Args:
           r: rotation angle(x, y, z) -- size = [N, 3]
       Returns:
           Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    batchsize = r.shape[0]
    r = F.expand_dims(F.expand_dims(F.clip(r, -np.pi, np.pi), -1), -1)
    zeros = xp.zeros((batchsize, 1, 1), dtype='f')
    ones = xp.ones((batchsize, 1, 1), dtype='f')
    cos_r = F.cos(r)
    sin_r = F.sin(r)
    rotz_1 = F.concat((cos_r[:, 2, :], -sin_r[:, 2, :], zeros), axis=2)
    rotz_2 = F.concat((-sin_r[:, 2, :], cos_r[:, 2, :], zeros), axis=2)
    rotz_3 = F.concat((zeros, zeros, ones), axis=2)
    zmat = F.concat((rotz_1, rotz_2, rotz_3), axis=1)

    roty_1 = F.concat((cos_r[:, 1, :], zeros, sin_r[:, 1, :]), axis=2)
    roty_2 = F.concat((zeros, ones, zeros), axis=2)
    roty_3 = F.concat((-sin_r[:, 1, :], zeros, cos_r[:, 1, :]), axis=2)
    ymat = F.concat((roty_1, roty_2, roty_3), axis=1)

    rotx_1 = F.concat((ones, zeros, zeros), axis=2)
    rotx_2 = F.concat((zeros, cos_r[:, 0, :], -sin_r[:, 0, :]), axis=2)
    rotx_3 = F.concat((zeros, sin_r[:, 0, :], cos_r[:, 0, :]), axis=2)
    xmat = F.concat((rotx_1, rotx_2, rotx_3), axis=1)

    rotMat = F.matmul(F.matmul(xmat, ymat), zmat)
    return rotMat

def pose_vec2mat(vec, xp):
    """Converts 6DoF parameters to transformation matrix

       Args:
           vec: 6DoF parameters in the order of rx, ry, rz, tx, ty, tz -- [N, 6]
       Returns:
           A transformation matrix -- [N, 4, 4]
    """
    r, t = vec[:, :3], vec[:, 3:]
    rot_mat = euler2mat(r, xp=xp)
    filler = xp.asarray([0.0, 0.0, 0.0, 1.0], dtype='f').reshape(1, 1, 4)
    batch_size = rot_mat.shape[0]
    filler = F.tile(filler, (batch_size, 1, 1))
    t = t.reshape(batch_size, 3, 1)
    transform_mat = F.dstack((rot_mat, t))
    transform_mat = F.hstack((transform_mat, filler))
    return transform_mat

def pixel2cam(depthes, pixel_coords, intrinsics, xp=np):
    N, _, H, W = depthes.shape
    cam_coords = F.matmul(F.batch_inv(intrinsics),
                          xp.broadcast_to(pixel_coords.reshape(1, 3, H*W),
                                          (N, 3, H*W)))
    depthes = F.broadcast_to(F.reshape(depthes, (N, 1, -1)), (N, 3, H*W))
    cam_coords = depthes * cam_coords
    cam_coords = F.concat((cam_coords, xp.ones((N, 1, H * W), dtype='f')), axis=1)
    return cam_coords.reshape(N, -1, H, W)

def cam2pixel(cam_coords, proj):
    N, _, H, W = cam_coords.shape
    cam_coords = F.reshape(cam_coords, (N, 4, -1))
    unnormalized_pixel_coords = F.matmul(proj, cam_coords)
    p_s_xy = unnormalized_pixel_coords[:, 0:2, :]
    p_s_xy /= F.broadcast_to(unnormalized_pixel_coords[:, 2:3, :], p_s_xy.shape) + 1e-10
    p_s_xy = F.reshape(p_s_xy, (N, 2, H, W))
    return p_s_xy

def generate_2dmeshgrid(H, W, xp=np):
    """Generate 2d meshgrid.
       Range of values is [-1, 1] because input range of bilinear interpolation
       is [-1, 1].
    """
    ys, xs = xp.meshgrid(
        xp.linspace(-1, 1, H, dtype=np.float32),
        xp.linspace(-1, 1, W, dtype=np.float32), indexing='ij',
        copy=False
    )
    coords = xp.concatenate(
        [xs[None], ys[None], xp.ones((1, H, W), dtype=np.float32)],
        axis=0)
    return coords
