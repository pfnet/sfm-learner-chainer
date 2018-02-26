#/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import numpy as np
import os
import random
from scipy.misc import imread
from collections import Counter
from tqdm import tqdm
import datetime
from chainer import functions as F
from chainer import dataset
import cv2

def load_as_float_norm(path):
    img = imread(path).astype(np.float32).transpose(2, 0, 1)
    return img / (255. * 0.5) - 1

class KittiOdometryEvaluation(dataset.DatasetMixin):

    """Dataset class for a task on `Kitti Depth Evaluation`_.

    Args:
        data_dir (string): Path to the dataset directory. The directory should
            contain at least three directories, :obj:`training`, `testing`
            and `ImageSets`.
    """
    def __init__(self, data_dir=None, test_files=None, gt_dir=None,
                 seq_len=3, height=128, width=416, seq_list=None):
        self.base_dir = data_dir
        self.seq_len = seq_len
        self.demi_len = (self.seq_len - 1) // 2
        self.height = height
        self.width = width
        self.seq_list = ['9', '10'] if seq_list is None else [str(seq_list)]
        self.data_list = []
        self.parse_data_list(test_files)
        self.imgs_file_list = []
        self.calib_dir_list = []
        self.gt_pose_list = []
        for data_list in self.data_list:
            self.read_scene_data(data_list)

        self.parse_gt_dirs(gt_dir)

    def parse_data_list(self, data_file):
        with open(data_file, 'r') as f:
            data_list = f.readlines()
        for d in data_list:
            if d.split(' ')[0] in self.seq_list:
                self.data_list += [d[:-1].split(' ')]

    def parse_gt_dirs(self, gt_dir):
        self.gt_files = glob.glob(os.path.join(gt_dir, "*.txt"))
        self.gt_files.sort()
        if not len(self.gt_files):
            print("There is not groudtruth data under {}".format(gt_dir))

    def read_scene_data(self, data_list):
        seq_id, date, drive, start, end = data_list
        data_dir = os.path.join(self.base_dir, date, drive)
        oxts_dir = os.path.join(data_dir, 'oxts')
        image_dir = os.path.join(data_dir, 'image_02/data')
        image_list = glob.glob(os.path.join(image_dir, '*.png'))
        image_list.sort()
        image_list = image_list[int(start):int(end) + 1]
        num_list = len(image_list)
        demi_len = (self.seq_len - 1) // 2
        src_iter = [i for i in range(-demi_len, demi_len+1) if i != 0]
        num_output = num_list - (self.seq_len - 1)

        for i in range(demi_len, num_list - demi_len):
            tgt_img_path = image_list[i]
            src_imgs_path = [image_list[i + si] for si in src_iter]
            self.calib_dir_list.append(os.path.join(self.base_dir, date))
            self.imgs_file_list.append([tgt_img_path, src_imgs_path])

    def __len__(self):
        return len(self.imgs_file_list)

    def get_example(self, i):
        if i % 100 == 0 and i != 0:
            percentage = i * 100 / len(self.imgs_file_list)
            print("Progress: {0:d}%".format(int(percentage)))
        calib_dir = self.calib_dir_list[i]
        imgs_path = self.imgs_file_list[i]
        tgt_img_path = imgs_path[0]
        src_imgs_path = imgs_path[1]
        tgt_img = load_as_float_norm(tgt_img_path)
        src_imgs = [load_as_float_norm(path) for path in src_imgs_path]
        gt_pose = read_file_list(self.gt_files[i])
        orig_shape = tgt_img.shape[:2]
        tgt_img = F.resize_images(tgt_img[None], (self.height, self.width)).data[0]
        src_imgs = F.resize_images(np.array(src_imgs, dtype='f'), (self.height, self.width)).data
        return tgt_img, src_imgs, [], gt_pose


def read_file_list(filename):
    """
    Reads a trajectory from a text file.

    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

    Input:
    filename -- File name

    Output:
    dict -- dictionary of (stamp,data) tuples

    """
    with open(filename, 'r') as f:
        data = f.read()
        lines = data.replace(","," ").replace("\t"," ").split("\n")
        data_list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
        return np.array([l for l in data_list if len(l)>1], dtype='f')
