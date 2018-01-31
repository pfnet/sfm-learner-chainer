#/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import numpy as np
import os
import random
from scipy.misc import imread

from chainer import dataset

def load_as_float_norm(path):
    img = imread(path).astype(np.float32).transpose(2, 0, 1)
    return img / (255. * 0.5) - 1

class KittiRawDataset(dataset.DatasetMixin):

    """Dataset class for a task on `Kitti Raw Dataset`_.

    Args:
        data_dir (string): Path to the dataset directory. The directory should
            contain at least three directories, :obj:`training`, `testing`
            and `ImageSets`.
        split ({'train', 'val'}): Select from dataset splits used in
            KiTTi Raw Dataset.
    """
    def __init__(self, data_dir=None, seq_len=3, split='train'):
        with open(os.path.join(data_dir, "{}.txt".format(split)), 'r') as f:
            dir_indexes = f.read().split('\n')

        if not dir_indexes[-1]:
            dir_indexes = dir_indexes[:-1]

        self.dir_pathes = [os.path.join(data_dir, index) for index in dir_indexes]
        self.seq_len = seq_len
        self.samples = self.crawl_folders()
        print('{} num sample'.format(split), len(self.samples))

    def crawl_folders(self):
        sequence_set = []
        demi_len = (self.seq_len - 1)//2
        for dir_path in self.dir_pathes:
            calib_path = os.path.join(dir_path, 'cam.txt')
            intrinsics = np.genfromtxt(calib_path, delimiter=',')
            intrinsics = intrinsics.astype(np.float32).reshape((3, 3))
            imgs = glob.glob(os.path.join(dir_path, '*.jpg'))
            imgs.sort()
            if len(imgs) < self.seq_len:
                continue
            for i in range(demi_len, len(imgs)-demi_len):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i],
                          'ref_imgs': []}
                for j in range(-demi_len, demi_len + 1):
                    if j != 0:
                        sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        return sequence_set

    def __len__(self):
        return len(self.samples)

    def save_img(self, tgt_img, ref_imgs):
        import cv2
        cv2.imwrite('tgt.png', tgt_img.transpose(1, 2, 0).astype('i'))
        cv2.imwrite('src1.png', ref_imgs[0].transpose(1, 2, 0).astype('i'))
        cv2.imwrite('src2.png', ref_imgs[1].transpose(1, 2, 0).astype('i'))

    def get_example(self, i):
        sample = self.samples[i]
        tgt_img = load_as_float_norm(sample['tgt'])
        ref_imgs = [load_as_float_norm(ref_img) for ref_img in sample['ref_imgs']]
        intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)
