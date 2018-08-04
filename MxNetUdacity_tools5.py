# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:06:54 2018

@author: Bird
下面进入产生优化指标的过程
"""
import mxnet as mx
from gluoncv.loss import _as_list
from mxnet import nd
from mxnet.gluon.loss import _reshape_like

class SSDMultiBoxLoss(mx.gluon.Block):
    """
    计算一个带有negative mining功能的 cls_losses, box_losses
    然后用一个系数将两个加起来生成 sum_losses

    Parameters
    ----------
    negative_mining_ratio : float, default is 3
        Ratio of negative vs. positive samples.
    rho : float, default is 1.0
        平滑L1的边界条件，为中文P381的 = 1/game^2
    lambd : float, default is 1.0
        两种loss的权重 

    最后的输出各维度与batch的维度是一样的
    """
    def __init__(self, negative_mining_ratio=3, rho=1.0, lambd=1.0, **kwargs):
        super(SSDMultiBoxLoss, self).__init__(**kwargs)
        self._negative_mining_ratio = max(0, negative_mining_ratio)
        self._rho = rho
        self._lambd = lambd

    def forward(self, cls_pred, box_pred, cls_target, box_target):
        #cls_pred:(32, 6132, 21) 这里面的分类，一个维度上仅表示一个分类
        #box_pred:(32, 6132, 4)
        #cls_target: (32, 6132) 注意，这里面的分类是从0开始顺序编码的
        #box_target(32, 6132, 4)
        
        #吧输入4个东西每一个都用一个list包裹起来
        """Compute loss in entire batch across devices."""
        # require results across different devices at this time
        cls_pred, box_pred, cls_target, box_target = [_as_list(x) \
            for x in (cls_pred, box_pred, cls_target, box_target)]
        # cross device reduction to obtain positive samples in entire batch
        num_pos = []
        for cp, bp, ct, bt in zip(*[cls_pred, box_pred, cls_target, box_target]):
            pos_samples = (ct > 0)
            num_pos.append(pos_samples.sum())
        num_pos_all = sum([p.asscalar() for p in num_pos])
        if num_pos_all < 1:
            # no positive samples found, return dummy losses
            return nd.zeros((1,)), nd.zeros((1,)), nd.zeros((1,))

        # compute element-wise cross entropy loss and sort, then perform negative mining
        cls_losses = []
        box_losses = []
        sum_losses = []
        for cp, bp, ct, bt in zip(*[cls_pred, box_pred, cls_target, box_target]):
            pred = nd.log_softmax(cp, axis=-1)
            pos = ct > 0
            cls_loss = -nd.pick(pred, ct, axis=-1, keepdims=False)
            rank = (cls_loss * (pos - 1)).argsort(axis=1).argsort(axis=1)
            hard_negative = rank < (pos.sum(axis=1) * self._negative_mining_ratio).expand_dims(-1)
            # mask out if not positive or negative
            cls_loss = nd.where((pos + hard_negative) > 0, cls_loss, nd.zeros_like(cls_loss))
            cls_losses.append(nd.sum(cls_loss, axis=0, exclude=True) / num_pos_all)

            bp = _reshape_like(nd, bp, bt)
            box_loss = nd.abs(bp - bt)
            box_loss = nd.where(box_loss > self._rho, box_loss - 0.5 * self._rho,
                                (0.5 / self._rho) * nd.square(box_loss))
            # box loss only apply to positive samples
            box_loss = box_loss * pos.expand_dims(axis=-1)
            box_losses.append(nd.sum(box_loss, axis=0, exclude=True) / num_pos_all)
            sum_losses.append(self._lambd * cls_losses[-1] + (1 - self._lambd) * box_losses[-1])

        return sum_losses, cls_losses, box_losses

#class SSDMultiBoxLoss(mx.gluon.Block):
# 这个是官方原版        
#    """
#    计算一个带有negative mining功能的 cls_losses, box_losses
#    然后用一个系数将两个加起来生成 sum_losses
#
#    Parameters
#    ----------
#    negative_mining_ratio : float, default is 3
#        Ratio of negative vs. positive samples.
#    rho : float, default is 1.0
#        平滑L1的边界条件，为中文P381的 = 1/game^2
#    lambd : float, default is 1.0
#        两种loss的权重 
#
#    最后的输出各维度与batch的维度是一样的
#    """
#    def __init__(self, negative_mining_ratio=3, rho=1.0, lambd=1.0, **kwargs):
#        super(SSDMultiBoxLoss, self).__init__(**kwargs)
#        self._negative_mining_ratio = max(0, negative_mining_ratio)
#        self._rho = rho
#        self._lambd = lambd
#
#    def forward(self, cls_pred, box_pred, cls_target, box_target):
#        #cls_pred:(32, 6132, 21) 这里面的分类，一个维度上仅表示一个分类
#        #box_pred:(32, 6132, 4)
#        #cls_target: (32, 6132) 注意，这里面的分类是从0开始顺序编码的
#        #box_target(32, 6132, 4)
#        
#        #吧输入4个东西每一个都用一个list包裹起来
#        """Compute loss in entire batch across devices."""
#        # require results across different devices at this time
#        cls_pred, box_pred, cls_target, box_target = [_as_list(x) \
#            for x in (cls_pred, box_pred, cls_target, box_target)]
#        # cross device reduction to obtain positive samples in entire batch
#        num_pos = []
#        for cp, bp, ct, bt in zip(*[cls_pred, box_pred, cls_target, box_target]):
#            pos_samples = (ct > 0)
#            num_pos.append(pos_samples.sum())
#        num_pos_all = sum([p.asscalar() for p in num_pos])
#        if num_pos_all < 1:
#            # no positive samples found, return dummy losses
#            return nd.zeros((1,)), nd.zeros((1,)), nd.zeros((1,))
#
#        # compute element-wise cross entropy loss and sort, then perform negative mining
#        cls_losses = []
#        box_losses = []
#        sum_losses = []
#        for cp, bp, ct, bt in zip(*[cls_pred, box_pred, cls_target, box_target]):
#            pred = nd.log_softmax(cp, axis=-1)
#            pos = ct > 0
#            cls_loss = -nd.pick(pred, ct, axis=-1, keepdims=False)
#            rank = (cls_loss * (pos - 1)).argsort(axis=1).argsort(axis=1)
#            hard_negative = rank < (pos.sum(axis=1) * self._negative_mining_ratio).expand_dims(-1)
#            # mask out if not positive or negative
#            cls_loss = nd.where((pos + hard_negative) > 0, cls_loss, nd.zeros_like(cls_loss))
#            cls_losses.append(nd.sum(cls_loss, axis=0, exclude=True) / num_pos_all)
#
#            bp = _reshape_like(nd, bp, bt)
#            box_loss = nd.abs(bp - bt)
#            box_loss = nd.where(box_loss > self._rho, box_loss - 0.5 * self._rho,
#                                (0.5 / self._rho) * nd.square(box_loss))
#            # box loss only apply to positive samples
#            box_loss = box_loss * pos.expand_dims(axis=-1)
#            box_losses.append(nd.sum(box_loss, axis=0, exclude=True) / num_pos_all)
#            sum_losses.append(cls_losses[-1] + self._lambd * box_losses[-1])
#
#        return sum_losses, cls_losses, box_losses

