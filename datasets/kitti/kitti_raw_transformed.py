#/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial
import time

import cv2 as cv
import numpy as np
from PIL import Image
import copy

from chainer import datasets
from datasets.kitti.kitti_raw_dataset import KittiRawDataset
from chainer import functions as F


def make_intrinsics_matrix(fx, fy, cx, cy):
    intrinsics = np.array([[fx, 0., cx],
                           [0., fy, cy],
                           [0., 0., 1.]], dtype='f')
    return intrinsics

def data_augmentation(tgt_img, src_imgs, intrinsics):
    """Data augmentation for training models.

       Args:
           tgt_img(ndarray): Shape is (3, H, W)
           src_img(list): Shape is [S, 3, H, W]
           intrinsics(ndarray): Shape is (3, 3)
    """
    # Random scaling
    def random_scaling(imgs, intrinsics):
        batch_size, _, in_h, in_w = imgs.shape
        scaling = np.random.uniform(1, 1.15, 2)
        x_scaling = scaling[0]
        y_scaling = scaling[1]
        out_h = int(in_h * y_scaling)
        out_w = int(in_w * x_scaling)
        imgs = F.resize_images(imgs, [out_h, out_w]).data
        fx = intrinsics[0,0] * x_scaling
        fy = intrinsics[1,1] * y_scaling
        cx = intrinsics[0,2] * x_scaling
        cy = intrinsics[1,2] * y_scaling
        intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)
        return imgs, intrinsics

    # Random cropping
    def random_cropping(imgs, intrinsics, out_h, out_w):
        batch_size, _, in_h, in_w = imgs.shape
        offset_y = int(np.random.uniform(1, 0, in_h - out_h + 1)[0])
        offset_x = int(np.random.uniform(1, 0, in_w - out_w + 1)[0])
        imgs = imgs[:, :, offset_y:offset_y+out_h, offset_x:offset_x+out_w]
        fx = intrinsics[0,0]
        fy = intrinsics[1,1]
        cx = intrinsics[0,2] - offset_x
        cy = intrinsics[1,2] - offset_y
        intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)
        return imgs, intrinsics

    # Random flip
    def random_flip(imgs, intrinsics):
        batch_size, _, in_h, in_w = imgs.shape
        if np.random.rand() < 0.5:
            imgs = imgs[:, :, :, ::-1]
            intrinsics[0, 2] = in_w - intrinsics[0, 2]
        return imgs, intrinsics

    _, out_h, out_w = tgt_img.shape
    imgs = np.concatenate((tgt_img[np.newaxis, :], src_imgs))
    imgs, intrinsics = random_scaling(imgs, intrinsics)
    imgs, intrinsics = random_cropping(imgs, intrinsics, out_h, out_w)
    imgs, intrinsics = random_flip(imgs, intrinsics)
    # im = tf.cast(im, dtype=tf.uint8)
    return imgs[0], [img for img in imgs[1:]], intrinsics

def get_multi_scale_intrinsics(intrinsics, n_scales):
    """Scale the intrinsics accordingly for each scale
       Args:
           intrinsics: Intrinsics for original image. Shape is (3, 3).
           n_scales(int): Number of scale.
       Returns:
           multi_intrinsics: Multi scale intrinsics.
    """
    multi_intrinsics = []
    for s in range(n_scales):
        fx = intrinsics[0, 0]/(2 ** s)
        fy = intrinsics[1, 1]/(2 ** s)
        cx = intrinsics[0, 2]/(2 ** s)
        cy = intrinsics[1, 2]/(2 ** s)
        intrinsics = np.array([[fx, 0., cx],
                               [0., fy, cy],
                               [0., 0., 1.]], dtype='f')
        multi_intrinsics.append(intrinsics)
    return multi_intrinsics

def _transform(inputs, n_scale=4, ):
    tgt_img, src_imgs, intrinsics, inv_intrinsics = inputs
    del inputs

    tgt_img, src_imgs, intrinsics = data_augmentation(tgt_img, src_imgs,
                                                      intrinsics)
    intrinsics = get_multi_scale_intrinsics(intrinsics, n_scale)
    return tgt_img, src_imgs, intrinsics, intrinsics


class KittiRawTransformed(datasets.TransformDataset):
    def __init__(self, data_dir=None, seq_len=3, split='train',
                 n_scale=4, ):
        self.d = KittiRawDataset(
            data_dir=data_dir, seq_len=seq_len, split=split)
        t = partial(
            _transform, n_scale=4, )
        super().__init__(self.d, t)
