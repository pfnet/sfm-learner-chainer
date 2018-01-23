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

class KittiDepthEvaluation(dataset.DatasetMixin):

    """Dataset class for a task on `Kitti Depth Evaluation`_.

    Args:
        data_dir (string): Path to the dataset directory. The directory should
            contain at least three directories, :obj:`training`, `testing`
            and `ImageSets`.
        split ({'train', 'val'}): Select from dataset splits used in
            KiTTi Raw Dataset.
    """
    def __init__(self, data_dir=None, test_files=None, seq_len=3, split='train'):
        with open(os.path.join(test_files), 'r') as f:
            file_pathes = f.read().split('\n')
        
        self.base_dir = data_dir
        self.file_pathes = file_pathes[:-1] if not file_pathes[-1] else file_pathes
        # self.file_pathes = [os.path.join(data_dir, path) for path in file_pathes]
        self.seq_len = seq_len
        self.read_scene_data()

    def read_scene_data(self):
        self.calib_dir_list, self.velo_file_list = [], []
        self.imgs_file_list, self.cams = [], []
        demi_len = (self.seq_len - 1) // 2
        file_iter = [i for i in range(-demi_len, demi_len+1) if i != 0]
        for file_path in self.file_pathes:
            date, scene, cam_id, _, index = file_path[:-4].split('/')
            scene_dir = os.path.join(self.base_dir, date, scene)
            img_dir = os.path.join(scene_dir, cam_id, 'data')
            imgs_path = [os.path.join(img_dir, '{:010d}.png'.format(int(index) + fi)) for fi in file_iter]
            velo_path = [os.path.join(scene_dir, 'velodyne_points/data/{}.bin'.format(index))]
            self.calib_dir_list.append(os.path.join(self.base_dir, date))
            self.velo_file_list.append(velo_path)
            self.imgs_file_list.append(imgs_path)
            self.cams.append(int(cam_id[-2:]))

    def crawl_folders(self):
        sequence_set = []
        demi_len = (self.seq_len - 1)//2
        for dir_path in self.r_pathes:
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
        return len(self.imgs_file_list)

    def get_example(self, i):
        sample = self.samples[i]
        tgt_img = load_as_float_norm(sample['tgt'])
        ref_imgs = [load_as_float_norm(ref_img) for ref_img in sample['ref_imgs']]
        intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)


def load_velodyne_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:,3] = 1
    return points


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def get_focal_length_baseline(calib_dir, cam=2):
    cam2cam = read_calib_file(calib_dir + 'calib_cam_to_cam.txt')
    P2_rect = cam2cam['P_rect_02'].reshape(3,4)
    P3_rect = cam2cam['P_rect_03'].reshape(3,4)

    # cam 2 is left of camera 0  -6cm
    # cam 3 is to the right  +54cm
    b2 = P2_rect[0,3] / -P2_rect[0,0]
    b3 = P3_rect[0,3] / -P3_rect[0,0]
    baseline = b3-b2

    if cam == 2:
        focal_length = P2_rect[0,0]
    elif cam == 3:
        focal_length = P3_rect[0,0]

    return focal_length, baseline


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def generate_depth_map(calib_dir, velo_file_name, im_shape, cam=2):
    # load calibration files
    cam2cam = read_calib_file(calib_dir/'calib_cam_to_cam.txt')
    velo2cam = read_calib_file(calib_dir/'calib_velo_to_cam.txt')
    velo2cam = np.hstack((velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3,3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3,4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_file_name)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,-1:]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:,0] < im_shape[1]) & (velo_pts_im[:,1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0
    return depth


def generate_mask(gt_depth, min_depth, max_depth):
    mask = np.logical_and(gt_depth > min_depth,
                          gt_depth < max_depth)
    # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
    gt_height, gt_width = gt_depth.shape
    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                     0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)

    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    return mask
