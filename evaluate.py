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

from kitti_eval.depth_util import print_depth_stats, compute_depth_errors
from kitti_eval.odom_util import print_odom_stats, compute_odom_errors
from kitti_eval.odom_util import convert_eval_format

def evaluate_odom(config, args):
    """Evaluate odometry prediction of sfm_learner by online"""
    test_data = load_dataset_test(config["dataset"])
    test_iter = create_iterator_test(test_data,
                                     config['iterator'])
    model = get_model(config["model"])
    devices = parse_devices(config['gpus'], config['updater']['name'])
    gpu_id = None if devices is None else devices['main']
    if devices:
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu(gpu_id)

    batchsize = config['iterator']['test_batchsize']

    index = 0
    num_data = len(test_iter.dataset)
    odom_error_all = []
    print("Start evaluation")
    for batch in test_iter:
        batch = chainer.dataset.concat_examples(batch, gpu_id)
        tgt_img, ref_imgs, _, gt_pose = batch
        _, pred_pose, _ = model.inference(tgt_img, ref_imgs,
                                          None, None, is_depth=False,
                                          is_pose=True, is_exp=False)
        pred_pose = chainer.cuda.to_cpu(F.concat(pred_pose, axis=0).data)
        pred_pose = np.insert(pred_pose, 2,  np.zeros((1, 6)), axis=0)
        gt_pose = chainer.cuda.to_cpu(gt_pose[0])
        pred_pose = convert_eval_format(pred_pose, gt_pose)
        odom_error = compute_odom_errors(pred_pose, gt_pose)
        if odom_error == False:
            continue
        odom_error_all.append(odom_error)
    odom_error_all = np.array(odom_error_all)
    print_odom_stats(odom_error_all)

def evaluate_depth(config, args):
    """Evaluate depth prediction of sfm_learner by online."""
    model = get_model(config["model"])
    devices = parse_devices(config['gpus'], config['updater']['name'])
    test_data = load_dataset_test(config["dataset"])
    test_iter = create_iterator_test(test_data,
                                     config['iterator'])

    gpu_id = None if devices is None else devices['main']
    if devices:
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu(gpu_id)

    min_depth = test_data.min_depth
    max_depth = test_data.max_depth
    batchsize = config['iterator']['test_batchsize']

    index = 0
    num_data = len(test_iter.dataset)
    sum_errors = np.array([0. for i in range(7)], dtype='f')
    for batch in test_iter:
        batch = chainer.dataset.concat_examples(batch, gpu_id)
        tgt_img, ref_imgs, _, gt_depth, mask = batch
        pred_depth, _, _ = model.inference(tgt_img, ref_imgs,
                                           None, None,
                                           is_depth=True, is_pose=False)
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
        sum_errors += compute_depth_errors(gt_depth, pred_depth) / num_data
    print_depth_stats(sum_errors)

def main():
    config, args = parse_args()
    if args.eval_mode == "depth":
        evaluate_depth(config, args)
    elif args.eval_mode == "odom":
        evaluate_odom(config, args)

if __name__ == '__main__':
    main()
