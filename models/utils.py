import numpy as np
from chainer import functions as F


def getG(r, xp):
    N = r.shape[0]
    G = xp.zeros(N, 3, 3)
    G[:, 1, 2] = r[:, 0]
    G[:, 2, 1] = -r[:, 0]
    G[:, 2, 0] = r[:, 1]
    G[:, 0, 2] = -r[:, 1]
    G[:, 0, 1] = r[:, 2]
    G[:, 1, 0] = -r[:, 2]


# rod rotation formula
def r2mat(r, xp):
    # Matrix exponential
    N = r.shape[0]
    I = xp.tile(xp.eye(3).reshape(1, 3, 3), (N, 1, 1))
    theta = r / F.sqrt(F.sum(r ** 2, axis=1, keepdims=True))
    G = getG(r, xp)
    return I + F.sin(theta) * G + (1 - F.cos(theta)) * F.batch_matmul(G, G)


def pose_vec2mat(vec, xp):
    r, t = vec[:, :3], vec[:, 3:]
    N = r.shape[0]
    R = r2mat(r, xp)
    zzzo = xp.tile(xp.asarray([0.0, 0.0, 0.0, 1.0], xp.float32).reshape(1, 1, 4), [N, 1, 1])
    Rt = F.concat([F.concat([R, t.reshape(N, 1, 3)], axis=2), zzzo], axis=1)
    return Rt


def generate_2dmeshgrid(H, W, xp=np):
    ys, xs = xp.meshgrid(
        xp.linspace(-1, 1, H, dtype=np.float32),
        xp.linspace(-1, 1, W, dtype=np.float32), indexing='ij',
        copy=False
    )
    coords = xp.concatenate(
        [xs[None], ys[None], xp.ones((1, H, W), dtype=np.float32)],
        axis=0)
    return coords
