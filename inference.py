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
from scipy.misc import imread
from kitti_eval.odom_util import convert_trajectory

from config_utils import *
import matplotlib as mpl
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0
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

def demo_depth_by_image(model, args, gpu_id):
    print("Inference for specified image")
    input_img = imread(args.img_path).astype(np.float32)
    input_img = cv2.resize(input_img, (args.width, args.height))
    img = input_img / (255. * 0.5) - 1
    img = img.transpose(2, 0, 1)[None, :]
    if gpu_id is not None:
        img = chainer.cuda.to_gpu(img, device=gpu_id)
    pred_depth, _, _ = model.inference(img, None, None, None,
                                       is_depth=True, is_pose=False)
    depth = chainer.cuda.to_cpu(pred_depth.data[0, 0])
    depth = normalize_depth_for_display(depth)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(input_img / 255)
    axes[1].imshow(depth)
    axes[0].axis('off')
    axes[1].axis('off')
    if args.save != -1:
        plt.savefig("output_{}.png".format(args.save),
                    bbox_inches="tight", pad_inches=0.0, transparent=True)
    plt.show()
    # cv2.imwrite("input.png", (input_img))
    # cv2.imwrite("depth.png", depth * 255 )
    print("Complete")

def demo_depth_by_dataset(model, config, gpu_id):
    test_data = load_dataset_test(config["dataset"])
    test_iter = create_iterator_test(test_data,
                                     config['iterator'])
    index = 0
    for batch in test_iter:
        input_img = batch[0][0].transpose(1, 2, 0)
        batch = chainer.dataset.concat_examples(batch, gpu_id)
        pred_depth, pred_pose, pred_mask = model.inference(*batch)
        depth = chainer.cuda.to_cpu(pred_depth.data[0, 0])
        depth = normalize_depth_for_display(depth)
        mask = chainer.cuda.to_cpu(pred_mask[0].data[0, 0])
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

def demo_odom_by_dataset(model, config, gpu_id):
    test_data = load_dataset_test(config["dataset"])
    test_iter = create_iterator_test(test_data,
                                     config['iterator'])
    index = 0
    num_data = len(test_iter.dataset)
    print("Start inference")
    base_pose = None
    scale_list = []
    for i, batch in enumerate(test_iter):
        if i % 4 != 0:
            continue
        batch = chainer.dataset.concat_examples(batch, gpu_id)
        tgt_img, ref_imgs, _, gt_pose = batch
        _, pred_pose, _ = model.inference(tgt_img, ref_imgs,
                                          None, None, is_depth=False,
                                          is_pose=True, is_exp=False)
        pred_pose = chainer.cuda.to_cpu(F.concat(pred_pose, axis=0).data)
        pred_pose = np.insert(pred_pose, 2, np.zeros((1, 6)), axis=0)
        gt_pose = chainer.cuda.to_cpu(gt_pose[0])
        pred_pose, orig_pose, base_pose = convert_trajectory(
                                              pred_pose, gt_pose,
                                              base_pose=base_pose)
        if i == 0:
            all_trajectory = pred_pose
            continue
        all_trajectory = np.concatenate((all_trajectory, pred_pose[1:, :]), axis=0)
    np.savetxt('test.txt', all_trajectory, delimiter=' ')

def visualize_odom(args, gt_file=None, pred_file=None):
    data_dict = {'gt_label': gt_file, 'pred_label': pred_file}
    for label, file_name in data_dict.items():
        if file_name:
            x = []
            z = []
            with open(file_name, 'r') as f:
                data = f.readlines()
            for d in data:
                d = d.split(" ")
                xyz = d[1:4]
                x.append(xyz[0])
                z.append(xyz[2])
            plt.plot(x, z, label=label)
            plt.legend()
    if args.save != -1:
        plt.savefig("result_{}.png".format(args.save))
    plt.show()

def demo_sfm_learner():
    """Demo sfm_learner."""
    config, args = parse_args()
    model = get_model(config["model"])
    devices = parse_devices(config['gpus'], config['updater']['name'])
    gpu_id = None if devices is None else devices['main']
    if devices:
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu(gpu_id)

    if args.mode == "depth":
        if args.img_path:
            demo_depth_by_image(model, args, gpu_id)
        else:
            demo_depth_by_dataset(model, config, gpu_id)
    elif args.mode == "odom":
        if args.gt_file or args.pred_file:
            visualize_odom(args, args.gt_file, args.pred_file)
        else:
            demo_odom_by_dataset(model, config, gpu_id)

def main():
    demo_sfm_learner()

if __name__ == '__main__':
    main()
