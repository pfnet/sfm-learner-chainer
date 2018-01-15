import chainer
from chainer.training import StandardUpdater
import numpy as np
from chainer import Variable
from chainer import functions as F
from transform import transform


class Updater(StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.disp_net = self.models['disp']
        self.pose = self.models['pose']
        self.coeff_smooth_reg = kwargs.pop('coeff_smooth_reg')
        self.coeff_exp_reg = kwargs.pop('coeff_exp_reg')
        if self.pose.n_sources != 2:
            raise ValueError("self.pose.n_sources should be 2. "
                             "self.pose.n_sources={}".format(self.pose.n_sources))
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        opt_disp = self.get_optimizer('opt_disp')
        opt_pose = self.get_optimizer('opt_pose')
        xp = opt_disp.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        _, H, W = batch[0][0].shape
        n_sources = len(batch[0][1])
        imgs_target = np.zeros((batchsize, 3, H, W), np.float32)
        imgs_source_list = [np.zeros((batchsize, 3, H, W), np.float32) for i in range(n_sources)]
        K = np.zeros((batchsize, 3, 3), np.float32)
        for i in range(batchsize):
            imgs_target[i] = batch[i][0]
            for i in range(n_sources):
                imgs_source_list[i][i] = batch[i][1][i]
            K[i] = batch[i][2]
        imgs_target = Variable(xp.asarray(imgs_target))
        imgs_sources = [Variable(xp.asarray(imgs_source_list[i]) for i in range(n_sources))]
        K = Variable(xp.asarray(K))

        disps = self.disp_net(imgs_target)
        depthes = [1. / d for d in disps]
        poses, mask_logits = self.pose(imgs_target, imgs_sources)

        pixel_loss = Variable(0)
        exp_loss = Variable(0)
        smooth_loss = Variable(0)
        for s in range(len(disps)):
            # Resize by bi-linear intp. (area intp. in the original inmplemenation)
            curr_imgs_target = F.resize_images(imgs_target, (H // (2 ** s), W // (2 ** s)))
            curr_imgs_sources = [F.resize_images(imgs_sources[i], (H // (2 ** s), W // (2 ** s)))
                                 for i in range(n_sources)]

            if self.coeff_smooth_reg > 0:
                smooth_loss += self.coeff_smooth_reg / (2 ** s) * \
                               self.compute_smooth_loss(disps[s])

            for i in range(n_sources):
                # Inverse warp the source image to the target image frame
                curr_proj_image = transform(
                    curr_imgs_sources[i],
                    F.squeeze(depthes[s], axis=1),
                    poses[:, i, :],
                    K[:, s, :, :])  # Why K has dimension for scale?
                curr_proj_error = F.absolute(curr_proj_image - curr_imgs_target)
                # Cross-entropy loss as regularization for the
                # explainability prediction
                if self.coeff_exp_reg > 0:
                    curr_exp_logits = F.slice(mask_logits[s],
                                              [0, i * 2, 0, 0],
                                              [-1, 2, -1, -1])
                    exp_loss += self.coeff_exp_reg * \
                                self.compute_exp_reg_loss(curr_exp_logits, xp)
                    curr_exp = F.softmax(curr_exp_logits)
                    pixel_loss += F.mean(curr_proj_error * curr_exp[:, 1:, :, :])
                else:
                    pixel_loss += F.mean(curr_proj_error)

        total_loss = pixel_loss + smooth_loss + exp_loss
        opt_disp.cleargrads()
        opt_pose.cleargrads()
        total_loss.backward()
        opt_disp.update()
        opt_pose.update()

    def compute_exp_reg_loss(self, pred, xp):
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
