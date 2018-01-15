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

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from models.transform import transform
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
        batchsize, n_sources, _, H, W = src_imgs.shape # tgt_img.shape
        stacked_src_imgs = self.xp.reshape(src_imgs, (batchsize, -1, H, W))
        pred_disps = self.disp_net(tgt_img)
        pred_depthes = [1 / d for d in pred_disps]
        pred_poses, pred_maskes = self.pose_net(tgt_img, stacked_src_imgs)
        smooth_loss, exp_loss, pixel_loss = 0, 0, 0
        n_scales = len(pred_depthes)
        for ns in range(n_scales):
            curr_img_size = (H // (2 ** ns), W // (2 ** ns))
            curr_tgt_img = F.resize_images(tgt_img, curr_img_size)
            curr_src_imgs = F.resize_images(stacked_src_imgs, curr_img_size)

            if self.smooth_reg:
                smooth_loss += self.smooth_reg / (2 ** ns) * \
                                   self.compute_smooth_loss(pred_disps[ns])

            for i in range(n_sources):
                # Inverse warp the source image to the target image frame
                i_channel = 3 * i
                curr_proj_image = transform(
                    curr_src_imgs[:, i_channel : (i_channel + 3)],
                    pred_depthes[ns],
                    pred_poses[i],
                    intrinsics)
                curr_proj_error = F.absolute(curr_proj_image - curr_tgt_img)
                # Cross-entropy loss as regularization for the
                # explainability prediction
                if self.exp_reg > 0:
                    pred_exp_logits = pred_maskes[ns][:, i*2:(i+1)*2, :, :]
                    exp_loss += self.exp_reg * \
                                    self.compute_exp_reg_loss(pred_exp_logits)
                    pred_exp = F.softmax(pred_exp_logits)[:, 1:, :, :]
                    pred_exp = F.broadcast_to(pred_exp, (batchsize, 3, H, W))
                    pixel_loss += F.mean(curr_proj_error * pred_exp)
                else:
                    pixel_loss += F.mean(curr_proj_error)

        total_loss = pixel_loss + smooth_loss + exp_loss
        chainer.report({'total_loss': total_loss}, self)
        chainer.report({'pixel_loss': pixel_loss}, self)
        chainer.report({'smooth_loss': smooth_loss}, self)
        chainer.report({'exp_loss': exp_loss}, self)
        return total_loss

    def compute_exp_reg_loss(self, pred):
        """Compute expalanation loss.

           Args:
               pred: Shape is (Batch, 2, H, W)
        """
        p_shape = pred.shape
        label = self.xp.ones((p_shape[0] * p_shape[2] * p_shape[3],), dtype='i')
        l = F.softmax_cross_entropy(
            F.reshape(pred, (-1, 2)), label)
        return F.mean(l)

    def compute_smooth_loss(self, pred_disp):
        """Compute smoothness loss for the predicted dpeth maps.
           L1 norm of the second-order gradients.

           Args:
               pred_disp: Shape is (Batch, 1, H, W)
        """
        def gradient(pred):
            D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
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
