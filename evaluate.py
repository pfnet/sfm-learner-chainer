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

import cv2
from config_utils import *
import matplotlib.pyplot as plt

chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"

from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))

def evaluate_depth():
    """Demo sfm_learner."""
    config, img_path = parse_args()
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

def main():
    evaluate_depth()

if __name__ == '__main__':
    main()
