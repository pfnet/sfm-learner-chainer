#/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import numpy as np
import os
import sys
import subprocess
import time
try:
    import matplotlib.pyplot as plt
except:
    pass
import cv2
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from models.transform import projective_inverse_warp
from models.pose_net import PoseNet
from models.disp_net import DispNet
from models.utils import *

def parse_dict(dic, key, value=None):
    return value if dic is None or not key in dic else dic[key]


class SFMLearner(chainer.Chain):

    """Sfm Learner original Implementation"""

    def __init__(self, config, pretrained_model=None):
        super(SFMLearner, self).__init__(
			pose_net = PoseNet(n_sources=config['seq_len'] - 1),
            disp_net = DispNet())

        self.smooth_reg = config['smooth_reg']
        self.exp_reg = config['exp_reg']
        self.ssim_rate = parse_dict(config, 'ssim_rate', 0.0)

        if pretrained_model['download']:
            if not os.path.exists(pretrained_model['download'].split("/")[-1]):
                subprocess.call(['wget', pretrained_model['download']])

        if pretrained_model['path']:
            chainer.serializers.load_npz(pretrained_model['path'], self)

    def __call__(self, tgt_img, src_imgs, intrinsics, inv_intrinsics):
        """
           Args:
               tgt_img: target image. Shape is (Batch, 3, H, W)
               src_imgs: source images. Shape is (Batch, ?, 3, H, W)
               intrinsics: Shape is (Batch, ?, 3, 3)
           Return:
               loss (Variable).
        """
        batchsize, n_sources, _, H, W = src_imgs.shape
        stacked_src_imgs = self.xp.reshape(src_imgs, (batchsize, -1, H, W))
        pred_disps = self.disp_net(tgt_img)
        pred_depthes = [1. / d for d in pred_disps]
        do_exp = self.exp_reg is not None and self.exp_reg > 0
        pred_poses, pred_maskes = self.pose_net(tgt_img, stacked_src_imgs,
                                                do_exp=do_exp)
        smooth_loss, exp_loss, pixel_loss = 0, 0, 0
        ssim_loss = 0
        n_scales = len(pred_depthes)
        start, stop = create_timer()
        sum_time = 0
        for ns in range(n_scales):
            curr_img_size = (H // (2 ** ns), W // (2 ** ns))
            curr_tgt_img = F.resize_images(tgt_img, curr_img_size).data
            curr_src_imgs = F.resize_images(stacked_src_imgs, curr_img_size).data

            # Smoothness regularization
            if self.smooth_reg:
                # smooth_loss += (self.smooth_reg / (2 ** ns)) * \
                #                     self.compute_smooth_loss(pred_disps[ns])
                smooth_loss += (self.smooth_reg / (2 ** ns)) * \
                                    self.compute_disp_smooth(curr_tgt_img,
                                                             pred_disps[ns])
            curr_pred_depthes = pred_depthes[ns]
            curr_pred_depthes = F.reshape(curr_pred_depthes, (batchsize, 1, -1))
            curr_pred_depthes = F.broadcast_to(curr_pred_depthes,
                                               (batchsize, 3, curr_pred_depthes.shape[2]))
            curr_intrinsics = intrinsics[:, ns]
            if self.exp_reg:
                curr_pred_mask = pred_maskes[ns]
            for i in range(n_sources):
                # Inverse warp the source image to the target image frame
                curr_proj_img = projective_inverse_warp(
                    curr_src_imgs[:, i*3:(i+1)*3],
                    curr_pred_depthes,
                    pred_poses[i],
                    curr_intrinsics)
                curr_proj_error = F.absolute(curr_proj_img - curr_tgt_img)
                mask = (curr_proj_img.data == 0).prod(1, keepdims=True).astype('bool')
                mask = self.xp.broadcast_to(mask, curr_proj_error.shape)
                curr_proj_error = F.where(mask,
                                          self.xp.zeros(curr_proj_error.shape, dtype='f'),
                                          curr_proj_error)
                # curr_proj_error *= (1 - (curr_proj_img.data == 0).prod(1, keepdims=True))
                # explainability regularization
                if self.exp_reg:
                    pred_exp_logits = curr_pred_mask[:, i:i+1, :, :]
                    exp_loss += self.exp_reg * \
                                    self.compute_exp_reg_loss(pred_exp_logits)
                    pred_exp = F.sigmoid(pred_exp_logits)
                    pred_exp = F.broadcast_to(pred_exp, (batchsize, 3, curr_img_size[0], curr_img_size[1]))
                    pixel_loss += F.mean(curr_proj_error * pred_exp)
                else:
                    pixel_loss += F.mean(curr_proj_error)
                    if self.ssim_rate:
                        ssim_error = self.compute_ssim(curr_proj_img, curr_tgt_img)
                        ssim_error *= (1 - mask)
                        ssim_loss += F.mean(ssim_loss)

        total_loss = (1 - self.ssim_rate) * pixel_loss + self.ssim_rate * ssim_loss + \
                         smooth_loss + exp_loss
        chainer.report({'total_loss': total_loss}, self)
        chainer.report({'pixel_loss': pixel_loss}, self)
        chainer.report({'smooth_loss': smooth_loss}, self)
        chainer.report({'exp_loss': exp_loss}, self)
        return total_loss

    def compute_ssim(self, x, y):
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_x = F.avg_pooling_2d(x, 3, 1, 1)
        mu_y = F.avg_pooling_2d(y, 3, 1, 1)

        sigma_x = F.avg_pooling_2d(x ** 2, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pooling_2d(y ** 2, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pooling_2d(x * y, 3, 1, 1) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y * c1) * (2 * sigma_xy + c2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)

        SSIM = SSIM_n / SSIM_d

        return F.clip((1 - SSIM) / 2, 0, 1)

    def compute_disp_smooth(self, img, pred_disp):
        def gradient(input_img):
            D_dy = input_img[:, :, 1:] - input_img[:, :, :-1]
            D_dx = input_img[:, :, :, 1:] - input_img[:, :, :, :-1]
            return D_dx, D_dy

        i_dx, i_dy = gradient(img)
        d_dx, d_dy = gradient(pred_disp)
        return F.mean(F.absolute(d_dx) * F.exp(-F.absolute(i_dx)) \
                      + F.absolute(d_dy) * F.exp(-F.absolute(i_dy)))

    def compute_exp_reg_loss(self, pred):
        """Compute expalanation loss.

           Args:
               pred: Shape is (Batch, 2, H, W)
        """
        p_shape = pred.shape
        label = self.xp.ones((p_shape[0] * p_shape[2] * p_shape[3],), dtype='i')
        l = F.sigmoid_cross_entropy(
            F.reshape(pred, (-1, )), label, reduce='no')
        return F.mean(l)

    def compute_smooth_loss(self, pred_disp):
        """Compute smoothness loss for the predicted dpeth maps.
           L1 norm of the second-order gradients.

           Args:
               pred_disp: Shape is (Batch, 1, H, W)
        """
        def gradient(pred):
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy

        dx, dy = gradient(pred_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        return F.mean(F.absolute(dx2)) + F.mean(F.absolute(dxdy)) \
               + F.mean(F.absolute(dydx)) + F.mean(F.absolute(dy2))

    def inference(self, tgt_img, src_imgs, intrinsics, inv_intrinsics,
                  is_depth=True, is_pose=True, is_exp=True):
        pred_depth, pred_pose, pred_mask = 0, 0, 0
        with chainer.using_config('train', False), \
                 chainer.function.no_backprop_mode():
            # #start, stop = create_timer()
            if is_depth:
                pred_depth = 1 / self.disp_net(tgt_img)[0]

            if is_pose:
                batchsize, n_sources, _, H, W = src_imgs.shape
                stacked_src_imgs = self.xp.reshape(src_imgs,
                                                   (batchsize, -1, H, W))
                pred_pose, pred_mask = self.pose_net(tgt_img,
                                                     stacked_src_imgs,
                                                     do_exp=is_exp)
            # #print_timer(#start, stop, sentence="Inference Time")
            return pred_depth, pred_pose, pred_mask
