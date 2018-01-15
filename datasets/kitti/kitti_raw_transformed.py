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

def make_intrinsics_matrix(fx, fy, cx, cy):
    # Assumes batch input
    batch_size = fx.get_shape().as_list()[0]
    zeros = tf.zeros_like(fx)
    r1 = tf.stack([fx, zeros, cx], axis=1)
    r2 = tf.stack([zeros, fy, cy], axis=1)
    r3 = tf.constant([0.,0.,1.], shape=[1, 3])
    r3 = tf.tile(r3, [batch_size, 1])
    intrinsics = tf.stack([r1, r2, r3], axis=1)
    return intrinsics

def data_augmentation(tgt_img, src_imgs, intrinsics, out_h, out_w):
    # Random scaling
    def random_scaling(im, intrinsics):
        batch_size, in_h, in_w, _ = im.get_shape().as_list()
        scaling = np.random.uniform(1, 1.15, 2)
        x_scaling = scaling[0]
        y_scaling = scaling[1]
        out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
        out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
        im = tf.image.resize_area(im, [out_h, out_w])
        fx = intrinsics[:,0,0] * x_scaling
        fy = intrinsics[:,1,1] * y_scaling
        cx = intrinsics[:,0,2] * x_scaling
        cy = intrinsics[:,1,2] * y_scaling
        intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)
        return im, intrinsics

    # Random cropping
    def random_cropping(im, intrinsics, out_h, out_w):
        # batch_size, in_h, in_w, _ = im.get_shape().as_list()
        batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
        offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
        offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
        im = tf.image.crop_to_bounding_box(
            im, offset_y, offset_x, out_h, out_w)
        fx = intrinsics[:,0,0]
        fy = intrinsics[:,1,1]
        cx = intrinsics[:,0,2] - tf.cast(offset_x, dtype=tf.float32)
        cy = intrinsics[:,1,2] - tf.cast(offset_y, dtype=tf.float32)
        intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)
        return im, intrinsics
    im, intrinsics = random_scaling(im, intrinsics)
    im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)
    im = tf.cast(im, dtype=tf.uint8)
    return im, intrinsics

def _transform(inputs, crop_size=(512, 512)):
    tgt_img, reg_imgs, intrinsics, inv_intrinsics = inputs
    del inputs

    # Global scaling
    if g_scale:
        scale = np.random.uniform(g_scale[0], g_scale[1], 4)
        pc *= scale
        places *= scale[:3]
        size *= scale[:3]

    # Flip
    if fliplr:
        if np.random.rand() > 0.5:
            pc[:, 1] = pc[:, 1] * -1
            places[:, 1] = places[:, 1] * -1
            rotates = rotates * -1

    return (feature_input, counter, indexes, gt_obj, gt_reg, gt_obj_for_reg,
            np.array([indexes.shape[0]]), np.array([n_no_empty]))


class KittiRawTransformed(datasets.TransformDataset):
    def __init__(self, data_dir="./", split="train", ignore_labels=True,
                 crop_size=(713, 713), color_sigma=None, g_scale=[0.5, 2.0],
                 resolution=None, x_range=None, y_range=None, z_range=None,
                 l_rotate=None, g_rotate=None, voxel_shape=None,
                 t=35, thres_t=3, norm_input=False, label_rotate=False,
                 anchor_size=(1.56, 1.6, 3.9), anchor_center=(-1.0, 0., 0.),
                 fliplr=False, n_class=19, scale_label=1):
        self.d = KittiRawDataset(
            data_dir, split, ignore_labels)
        t = partial(
            _transform, crop_size=crop_size, g_scale=g_scale)
        super().__init__(self.d, t)
