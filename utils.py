import numpy as np


def euler2mat(z, y, x):
    raise NotImplementedError


def pose_vec2mat(vec):
    raise NotImplementedError


def generate_meshgrid(H, W, xp=np):
    ys, xs = xp.meshgrid(
        xp.linspace(-1, 1, H, dtype=np.float32),
        xp.linspace(-1, 1, W, dtype=np.float32), indexing='ij',
        copy=False
    )
    coords = xp.concatenate(
        [xs[None], ys[None], xp.ones((1, H, W), dtype=np.float32)],
        axis=0)
    return coords
