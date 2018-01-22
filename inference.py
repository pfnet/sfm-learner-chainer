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

from config_utils import *

chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"

from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))

def demo_sfm_learner():
    """Demo sfm_learner."""
    config = parse_args()
    model = get_model(config["model"])
    devices = parse_devices(config['gpus'], config['updater']['name'])
    test_data = load_dataset_test(config["dataset"])
    test_iter = create_iterator_test(test_data,
                                     config['iterator'])
    model.to_gpu(devices['main'])

    dataset_config = config['dataset']['test']['args']
    index = 0
    for batch in test_iter:
        print(test_iter, test_data)
        model.predict(*batch, dataset_config)
        index += 1

def main():
    demo_sfm_leaner()

if __name__ == '__main__':
    main()
