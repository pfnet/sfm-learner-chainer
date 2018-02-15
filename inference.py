#!/usr/env/bin python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import sys
import subprocess
import os
import yaml

import chainer
from chainer import cuda, optimizers, serializers
from chainer import training
from chainer import functions as F

import cv2
from config_utils import *
import matplotlib.pyplot as plt

chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"

from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))

def normalize_depth_for_display(disp, pc=95, crop_percent=0, normalizer=None,
                                cmap='gray'):
    """Convert disparity images to depth images."""
    depth = 1./(disp + 1e-6)
    if normalizer is not None:
        depth = depth/normalizer
    else:
        depth = depth/(np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1-crop_percent))
    depth = depth[:keep_H]
    depth = depth
    return depth

def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img

def demo_sfm_learner():
    """Demo sfm_learner."""
    config, args = parse_args()
    model = get_model(config["model"])
    devices = parse_devices(config['gpus'], config['updater']['name'])
    test_data = load_dataset_test(config["dataset"])
    test_iter = create_iterator_test(test_data,
                                     config['iterator'])
    model.to_gpu(devices['main'])

    dataset_config = config['dataset']['test']['args']
    index = 0
    for batch in test_iter:
        input_img = batch[0][0].transpose(1, 2, 0)
        batch = chainer.dataset.concat_examples(batch, devices['main'])
        pred_depth, pred_pose, pred_mask = model.inference(*batch)
        depth = chainer.cuda.to_cpu(pred_depth.data[0, 0])
        depth = normalize_depth_for_display(depth)
        mask = chainer.cuda.to_cpu(pred_mask.data[0, 0])
        cv2.imwrite("input_{}.png".format(index), (input_img + 1) / 2 * 255)
        cv2.imwrite("depth_{}.png".format(index), depth * 255 )
        per = np.percentile(mask, 99)
        mask = mask * (mask < per)
        mask_min = mask.min()
        mask_max = mask.max()
        mask = (1 - (mask - mask_min) / mask_max) * 255
        cv2.imwrite("exp_{}.png".format(index), mask)
        print(index)
        index += 1

def main():
    demo_sfm_learner()

if __name__ == '__main__':
    main()
