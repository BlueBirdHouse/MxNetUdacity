# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:57:52 2018

@author: Bird
"""
from mxnet.gluon import HybridBlock
from mxnet import nd
#from gluoncv.nn.coder import NormalizedBoxCenterDecoder

class BBoxCornerToCenter(HybridBlock):
    """Convert corner boxes to center boxes.
    Corner boxes are encoded as (xmin, ymin, xmax, ymax)
    Center boxes are encoded as (center_x, center_y, width, height)

    Parameters
    ----------
    split : bool
        Whether split boxes to individual elements after processing.
    axis : int, default is -1
        Effective axis of the bounding box. Default is -1(the last dimension).

    Returns
    -------
     A BxNx4 NDArray if split is False, or 4 BxNx1 NDArray if split is True
    """
    def __init__(self, axis=-1, split=False):
        super(BBoxCornerToCenter, self).__init__()
        self._split = split
        self._axis = axis

    def hybrid_forward(self, F, x):
        xmin, ymin, xmax, ymax = F.split(x, axis=self._axis, num_outputs=4)
        width = xmax - xmin
        height = ymax - ymin
        x = xmin + width / 2
        y = ymin + height / 2
        if not self._split:
            return F.concat(x, y, width, height, dim=self._axis)
        else:
            return x, y, width, height

class NormalizedBoxCenterDecoder(HybridBlock):
    """
    #将网络的预测输出转换为xyxy结构的绝对坐标系统
    输入x = box_preds(1x9504x4), anchors(1x9504x4)两个网络输出。
    需要与 NormalizedBoxCenterEncoder 使用相同的参数 `stds` 才能发挥作用。
    使用xyxy格式输出猫框 `x_{min}, y_{min}, x_{max}, y_{max}`.
    stds看上去是为了调节网络输出范围的修正参数
    Parameters
    ----------
    stds : array-like of size 4
        Std value to be divided from encoded values, default is (0.1, 0.1, 0.2, 0.2).

    """
    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2),means=(0., 0., 0., 0.),
                 convert_anchor=False, clip=None):
        super(NormalizedBoxCenterDecoder, self).__init__()
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        self._stds = stds
        self._means = means
        self._clip = clip
        if convert_anchor:
            self.corner_to_center = BBoxCornerToCenter(split=True)
        else:
            self.corner_to_center = None

    def hybrid_forward(self, F, x, anchors):        
        if self.corner_to_center is not None:
            a = self.corner_to_center(anchors)
        else:
            #anchors按照最后一个坐标分，分四份。就是把每一个维度都分开了
            a = anchors.split(axis=-1, num_outputs=4)
        #网络预测的位置也按照维度分开
        p = F.split(x, axis=-1, num_outputs=4)
        #这个求法与英文文章P4/15上的公式的逆运算
        #后面多出来的2是为了转换为xyxy格式以后不用写/2这个部分，代码比较好看
        ox = F.broadcast_add(F.broadcast_mul(p[0] * self._stds[0] + self._means[0], a[2]), a[0])
        oy = F.broadcast_add(F.broadcast_mul(p[1] * self._stds[1] + self._means[1], a[3]), a[1])
        tw = F.exp(p[2] * self._stds[2] + self._means[2])
        th = F.exp(p[3] * self._stds[3] + self._means[2])
        if self._clip:
            tw = F.minimum(tw, self._clip)
            th = F.minimum(th, self._clip)
        ow = F.broadcast_mul(tw, a[2]) / 2
        oh = F.broadcast_mul(th, a[3]) / 2
        return F.concat(ox - ow, oy - oh, ox + ow, oy + oh, dim=-1)



class MultiPerClassDecoder(HybridBlock):
    """Decode classification results.
    输出大于阈值的分类标号和致信度。
    这一层的输入需要自己通过softmax函数
    softmax(cls_preds, axis=-1)
    必须与`MultiClassEncoder`一同工作。 
    
    This version is different from
    与py:class:`gluoncv.nn.coder.MultiClassDecoder`不同的是，可以输出对所有类别的
    评分，而不会仅仅输出最大的评分

    Parameters
    ----------
    num_class : int
        包含背景在内的分类个数
    axis : int
        Axis of class-wise results.
    thresh : float
        分类阈值。单位是对应通过softmax函数以后得到的。
        小于阈值的得分为0， 对应的分类标记为-1. 
        

    """
    def __init__(self, num_class, axis=-1, thresh=0.01):
        super(MultiPerClassDecoder, self).__init__()
        self._fg_class = num_class - 1 #得出实际识别物体的类别，因为没有人要框住背景
        self._axis = axis
        self._thresh = thresh

    def hybrid_forward(self, F, x):
        #x=1x9504x21
        #切掉第0位关于背景的评分
        scores = x.slice_axis(axis=self._axis, begin=1, end=None)  # b x N x fg_class
        #生成一个1x9504x1的模板
        template = F.zeros_like(x.slice_axis(axis=-1, begin=0, end=1))
        cls_ids = []
        for i in range(self._fg_class):
            #cls_ids里面是一系列的1x9504x1的类别编码，从0-19
            cls_ids.append(template + i)  # b x N x 1
        #重新生成1x9504x20的矩阵，每一行从0到19.这里应该有更好的方法吧。
        cls_id = F.concat(*cls_ids, dim=-1)  # b x N x fg_class
        #执行mask功能
        mask = scores > self._thresh
        cls_id = F.where(mask, cls_id, F.ones_like(cls_id) * -1)
        scores = F.where(mask, scores, F.zeros_like(scores))
        return cls_id, scores
