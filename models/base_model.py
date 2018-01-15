import chainer
from chainer.training import StandardUpdater
import numpy as np
from chainer import Variable
from chainer import functions as F
from models.transform import transform

#/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division

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

import chainer
import chainer.functions as F
import chainer.links as L
from models.pose_net import PoseNet
from models.disp_net import DispNet

def create_timer():
    start = chainer.cuda.Event()
    stop = chainer.cuda.Event()
    start.synchronize()
    start.record()
    return start, stop

def print_timer(start, stop, sentence="Time"):
    stop.record()
    stop.synchronize()
    elapsed_time = chainer.cuda.cupy.cuda.get_elapsed_time(
                           start, stop) / 1000
    print(sentence, elapsed_time)
    return elapsed_time


class SFMLearner(chainer.Chain):

    """Sfm Learner original Implementation"""

    def __init__(self, config, pretrained_model=None):
        super(SFMLearner, self).__init__(
			pose_net = PoseNet(),
            disp_net = DispNet())

        self.smooth_reg = config['smooth_reg']
        self.exp_reg = config['exp_reg']

        if pretrained_model['download']:
            if not os.path.exists(pretrained_model['download'].split("/")[-1]):
                subprocess.call(['wget', pretrained_model['download']])

        if pretrained_model['path']:
            chainer.serializers.load_npz(pretrained_model['path'], self)

    def __call__(self, tgt_img, src_imgs, intrinsics, inv_intrinsics):
        """
           Args:
               tgt_img: target image. Shape is (Batch, 3, H, W)
               src_imgs: source images. Shape is (Batch, ?, H, W)
           Return:
               loss (Variable).
        """
        batchsize, _, H, W = tgt_img.shape
        pred_disps = self.disp_net(tgt_img)
        pred_depthes = [1 / d for d in pred_disps]
        pred_pose, pred_mask = self.pose(tgt_img, src_imgs)
        smooth_loss, exp_loss, pixel_loss = 0, 0, 0
        for d in range(len(pred_depthes)):
            curr_img_size = (H // (2 ** d), W // (2 ** d))
            curr_tgt_img = F.resize_images(tgt_img, curr_img_size)
            curr_src_imgs = F.resize_images(src_imgs, curr_img_size)

            if self.smooth_reg:
                smooth_loss += self.smooth_loss / (2 ** d) * \
                                   self.compute_smooth_loss(pred_disps[d])

            for i in range(n_sources):
                # Inverse warp the source image to the target image frame
                curr_proj_image = transform(
                    curr_src_imgs[i],
                    F.squeeze(pred_depthes[d], axis=1),
                    poses[:, i, :],
                    K[:, d, :, :])  # Why K has dimension for scale?
                curr_proj_error = F.absolute(curr_proj_image - curr_tgt_img)
                # Cross-entropy loss as regularization for the
                # explainability prediction
                if self.exp_reg > 0:
                    curr_exp_logits = F.slice(mask_logits[d],
                                              [0, i * 2, 0, 0],
                                              [-1, 2, -1, -1])
                    exp_loss += self.exp_reg * \
                                    self.compute_exp_reg_loss(curr_exp_logits)
                    curr_exp = F.softmax(curr_exp_logits)
                    pixel_loss += F.mean(curr_proj_error * curr_exp[:, 1:, :, :])
                else:
                    pixel_loss += F.mean(curr_proj_error)

        total_loss = pixel_loss + smooth_loss + exp_loss
        chainer.report({'total_loss': total_loss}, self)
        chainer.report({'pixel_loss': pixel_loss}, self)
        chainer.report({'smooth_loss': smooth_loss}, self)
        chainer.report({'exp_loss': exp_loss}, self)
        return total_loss

    def compute_exp_reg_loss(self, pred):
        tmp = np.array([0, 1], dtype=np.float32).reshape(1, 2, 1, 1)
        ref_exp_mask = np.tile(tmp, (pred.shape[0], 1, pred.shape[2], pred.shape[3]))
        ref_exp_mask = xp.asarray(ref_exp_mask, dtype=xp.float32)
        l = F.softmax_cross_entropy(
            F.reshape(pred, (-1, 2)), F.reshape(ref_exp_mask, (-1, 2))
        )
        return F.mean(l)

    def compute_smooth_loss(self, pred_disp):
        def gradient(pred):
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            return D_dx, D_dy

        dx, dy = gradient(pred_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        return F.mean(F.absolute(dx2)) + F.mean(F.absolute(dxdy)) \
               + F.mean(F.absolute(dydx)) + F.mean(F.absolute(dy2))

    def inference(self, x, counter, indexes, batch, n_no_empty,
                  config=None, thres_prob=0.996,
                  anchor_size=None, anchor_center=None, anchor=None):
        with chainer.using_config('train', False), \
                 chainer.function.no_backprop_mode():
            sum_time = 0
            start, stop = create_timer()
            x = self.feature_net(x)
            sum_time += print_timer(start, stop, sentence="feature net")
            start, stop = create_timer()
            x = feature_to_voxel(x, indexes, self.k, self.d, self.h, self.w, batch)
            sum_time += print_timer(start, stop, sentence="feature_to_voxel")
            start, stop = create_timer()
            x = self.middle_conv(x)
            sum_time += print_timer(start, stop, sentence="middle_conv")
            start, stop = create_timer()
            pred_prob, pred_reg = self.rpn(x)
            sum_time += print_timer(start, stop, sentence="rpn")
            print("## Sum of execution time: ", sum_time)
            s = time.time()
            pred_reg = self.xp.transpose(pred_reg, (0, 2, 3, 1)).data[0]
            pred_prob = pred_prob[0, 0].data
            candidate = F.sigmoid(pred_prob).data > thres_prob
            pred_prob = pred_prob[candidate]
            pred_reg = pred_reg[candidate]
            pred_prob = chainer.cuda.to_cpu(pred_prob)
            pred_reg = chainer.cuda.to_cpu(pred_reg)
            candidate = chainer.cuda.to_cpu(candidate)
            anchor = anchor[candidate]
            pred_reg = self.decoder(pred_reg, anchor, anchor_size, xp=np)
            # print(pred_reg[:, :3])
            sort_index = np.argsort(pred_prob)[::-1]
            pred_reg = pred_reg[sort_index]
            pred_prob = pred_prob[sort_index]
            result_index = nms_3d(pred_reg,
                                  pred_prob,
                                  thres_prob)
            print("Post-processing", time.time() - s)
            # print(sort_index[result_index])
        return pred_reg[result_index], pred_prob[result_index]


    def predict(self, x, counter, indexes, gt_prob, gt_reg, batch,
                n_no_empty, config=None):
        with chainer.using_config('train', False), \
                 chainer.function.no_backprop_mode():
            sum_time = 0
            start, stop = create_timer()
            x = self.feature_net(x)
            sum_time += print_timer(start, stop, sentence="feature net")
            start, stop = create_timer()
            x = feature_to_voxel(x, indexes, self.k, self.d, self.h, self.w, batch)
            sum_time += print_timer(start, stop, sentence="feature_to_voxel")
            self.viz_input(x)
            start, stop = create_timer()
            x = self.middle_conv(x)
            sum_time += print_timer(start, stop, sentence="middle_conv")
            start, stop = create_timer()
            pred_prob, pred_reg = self.rpn(x)
            sum_time += print_timer(start, stop, sentence="rpn")
            print("## Sum of execution time: ", sum_time)
            if config is not None:
                print("#####   Visualize   #####")
                self.visualize(pred_reg, gt_reg, pred_prob, gt_prob, **config)
            else:
                print("##### Calc accuracy #####")
                return self.calc_accuracy(pred_reg, gt_reg, pred_prob, gt_prob)
