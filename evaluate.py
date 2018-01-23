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

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return np.array([abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3], dtype='f')

def print_stats(sum_errors):
    error_names = ['abs_rel','sq_rel','rms','log_rms','a1','a2','a3']
    print("Results with scale factor determined by GT/prediction ratio (like the original paper) : ")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*sum_errors))

def evaluate_depth():
    """Evaluate depth prediction of sfm_learner."""
    config, img_path = parse_args()
    model = get_model(config["model"])
    devices = parse_devices(config['gpus'], config['updater']['name'])
    test_data = load_dataset_test(config["dataset"])
    test_iter = create_iterator_test(test_data,
                                     config['iterator'])
    model.to_gpu(devices['main'])
    min_depth = test_data.min_depth
    max_depth = test_data.max_depth
    batchsize = config['iterator']['test_batchsize']

    index = 0
    num_data = len(test_iter.dataset)
    sum_errors = np.array([0. for i in range(7)], dtype='f')
    for batch in test_iter:
        batch = chainer.dataset.concat_examples(batch, devices['main'])
        tgt_img, ref_imgs, intrinsics, gt_depth, mask = batch
        pred_depth, pred_pose, pred_exp = model.inference(tgt_img,
                                                           ref_imgs,
                                                           intrinsics, None)
        batchsize = pred_depth.shape[0]
        pred_depth = F.resize_images(pred_depth, gt_depth.shape[1:]).data
        pred_depth = F.clip(pred_depth, min_depth, max_depth).data[:, 0]
        pred_depth = chainer.cuda.to_cpu(pred_depth)
        mask = chainer.cuda.to_cpu(mask)
        gt_depth = chainer.cuda.to_cpu(gt_depth)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        scale_factor = np.median(gt_depth) / np.median(pred_depth)
        pred_depth *= scale_factor
        sum_errors += compute_errors(gt_depth, pred_depth) / num_data
    print_stats(sum_errors)

def main():
    evaluate_depth()

if __name__ == '__main__':
    main()
