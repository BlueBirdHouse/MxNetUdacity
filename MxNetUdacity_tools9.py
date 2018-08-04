# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 14:16:47 2018

@author: Bird

"""

#%% 
from gluoncv.data.transforms import experimental
from gluoncv.data.transforms import image as timage
from gluoncv.data.transforms import bbox as tbbox
from gluoncv.nn.matcher import CompositeMatcher, BipartiteMatcher, MaximumMatcher
from gluoncv.nn.sampler import OHEMSampler, NaiveSampler
from gluoncv.nn.coder import MultiClassEncoder, NormalizedBoxCenterEncoder
from gluoncv.nn.bbox import BBoxCenterToCorner

import numpy as np

import mxnet as mx

from mxnet import nd
from mxnet.gluon import Block


class SSDTargetGenerator(Block):
    """
    根据label生成可以与网络输出直接比较计算误差的网络训练目标。mbox_loss完成实际的误差计算
    相当于中文教材的training_targets
    回答了：这个Anchor是什么的问题。是背景还是某个东西
    Parameters
    ----------
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    neg_thresh : float
        IOU overlap threshold for negative mining, default is 0.5.
    negative_mining_ratio : float
        Ratio of hard vs positive for negative mining.
    stds : (0.1, 0.1, 0.2, 0.2)这个会对框做一个加密，虽然真实的目的应该是对
        框预测的值做数值修正，让其更好的能被神经网络拟合。但是，如果后面的部分是自己写得，
        不知道这个事情， 就会无法解析网络输出的结果。 
    """
    def __init__(self, iou_thresh=0.5, neg_thresh=0.5, negative_mining_ratio=3,
                 stds=(0.1, 0.1, 0.2, 0.2), **kwargs):
        super(SSDTargetGenerator, self).__init__(**kwargs)
        self._matcher = CompositeMatcher([BipartiteMatcher(), MaximumMatcher(iou_thresh)])
        if negative_mining_ratio > 0:
            self._sampler = OHEMSampler(negative_mining_ratio, thresh=neg_thresh)
            self._use_negative_sampling = True
        else:
            #实际使用的是这个模式
            self._sampler = NaiveSampler()
            self._use_negative_sampling = False
        self._cls_encoder = MultiClassEncoder()
        self._box_encoder = NormalizedBoxCenterEncoder(stds=stds)
        self._center_to_corner = BBoxCenterToCorner(split=False)

    # pylint: disable=arguments-differ
    def forward(self, anchors, cls_preds, gt_boxes, gt_ids):
        """Generate training targets."""
        
        #anchors，网络生成的猫框(1, N, 4)。由于针对每一个图片，anchors肯定是一样的，所以第一个维度是1，而不是当前batch的个数
        #网络输出的猫框是中心+长宽模式
        #后面两个的第一维度是1，那是因为在SSDDefaultTrainTransform里面，只针对一个图片做变换
        #在DataLoader里面，需要将这些数据叠加起来，但是，anchors的第一个维度仍然是1
        #cls_preds是空的
        #gt_bboxes(label)里面是一个图像上的box位置，实际像素点位 (1,当前图片里面的框框个数,4)
        #gt_ids(label)里面是对应图像上的label，(1,当前图片里面的框框个数,1)
        #data: (2, 3, 512, 512) label: (2, 6, 5)
        
        
        '''
        #这个仅仅基于Anchors的转换处理，回答了：这个Anchor是什么的问题。是背景还是某个东西
        #针对每一个Anchors的位置，都给出一个cls_targets和box_targets
        #但是，如果某些Anchors对cls_targets的IOU太小的话，对应的cls_targets和box_targets位置就会被填充0
        #这表示他们是背景。换句话说，实际使用的类别编码都被+1，因为增加了背景
        #同时，box的位置别转换成了英文教材4/15上面那种偏差的形式（利用NormalizedBoxCenterEncoder做的）
        #与中文教材P378使用的MultiBoxTarget相比，仅完成了任务1.不考虑保留一些最不确信是背景的框。
        #认为网络分类是完全准确的。 
        #返回值是train_dataset[0] = 3x512x512, 6132, 6132x4
        '''
        #去掉anchors前面的多余坐标系1，然后变为两点模式。注意，网络的输出是长宽模式
        anchors = self._center_to_corner(anchors.reshape((-1, 4)))
        #集合A对集合B中的每一个元素都可以计算IOU，所以计算结果是：
        #anchors个*1*当前图片里面的框框个数 ->(transpose)(1*anchors个*当前图片里面的框框个数)
        ious = nd.transpose(nd.contrib.box_iou(anchors, gt_boxes), (1, 0, 2))
        #找到IOU超过阈值的。这里需要利用专门的函数处理，是因为要引入负例采样。
        #找一些快要错的部分作为负例专门训练。所以有正例，负例和忽略三个选项
        matches = self._matcher(ious)
        if self._use_negative_sampling:
            samples = self._sampler(matches, cls_preds, ious)
        else:
            #实际使用的是这里
            samples = self._sampler(matches)
        #从这里可以看到，分类训练和框框训练都受到了IOU的影响。
        #cls_targets里面0要表示背景，所以所有的分类编号在这里+1
        cls_targets = self._cls_encoder(samples, matches, gt_ids)
        #除了完成英文书P4/15里面的公式以外，还要做一般化，及用到了stds
        box_targets, box_masks = self._box_encoder(samples, matches, anchors, gt_boxes)
        #返回维度是：
        #cls_targets = 1*喵框个数 里面的分类标记+1，0表示背景或者是不需要学习的
        #box_targets = 1*喵框个数*4, 需要学习的偏差量，不需要学习的是0
        #box_masks = 1*喵框个数*4,没有被选中的是0
        #返回量已经应用了box_masks，这就是实际中不使用masks的原因。
        return cls_targets, box_targets, box_masks



class SSDDefaultTrainTransform(object):
    """
    利用内部算法，对图片做各种随机变化，从而增强模型的鲁棒性
    这里有一个魔术，如果初始化的时候通入了anchors，那么会执行label对齐，以便适应模型训练。
    如果不通入anchors，那么label仍然是根据每一个图片内框的不同而不同。

    """
    def __init__(self, width, height, anchors=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), iou_thresh=0.5, box_norm=(0.1, 0.1, 0.2, 0.2),
                 **kwargs):
        '''
        输出的图片，连同对应的框标记，都会做相应的变化，最后图片的大小是：width，
        width :int
        height : int
        
        anchors : 由网络输出的猫框，利用这个信息为后来的mbox_loss计算网络损失.是中心+长宽模式
    
        Anchors generated from SSD networks, the shape must be ``(1, N, 4)``.
        由于针对每一个图片，anchors肯定是一样的，所以第一个维度是1，而不是当前batch的个数
        ``N`` 是对应这个图片应该生成多少个猫框，取决于作者的设计。
        
        模型归一化用的均值和方差，由于使用imageNet训练以后的检测头，所以应该与imageNet保持一致
        mean : array-like of size 3，从图片上减去的。Default is [0.485, 0.456, 0.406].
        std : array-like of size 3，从图片上除去。Default is [0.229, 0.224, 0.225].
        
        
        iou_thresh : float。判断两个框是否重合的阈值。IOU=重合面积占两个框面积之和 
        IOU overlap threshold for maximum matching, default is 0.5.
    
        box_norm : (0.1, 0.1, 0.2, 0.2)这个会对框做一个加密，虽然真实的目的应该是对
        框预测的值做数值修正，让其更好的能被神经网络拟合。但是，如果后面的部分是自己写得，
        不知道这个事情， 就会无法解析网络输出的结果。 
        '''
        
        self._width = width
        self._height = height
        self._anchors = anchors
        self._mean = mean
        self._std = std
        #如果不设定anchors，说明肯定不是用在训练过程中，这里直接返回
        if anchors is None:
            return

        # since we do not have predictions yet, so we ignore sampling here
        self._target_generator = SSDTargetGenerator(
            iou_thresh=iou_thresh, stds=box_norm, negative_mining_ratio=-1, **kwargs)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        # random color jittering
        img = experimental.image.random_color_distort(src)

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, expand = timage.random_expand(img, fill=[m * 255 for m in self._mean])
            bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
        else:
            img, bbox = img, label

        # random cropping
        h, w, _ = img.shape
        bbox, crop = experimental.bbox.random_crop_with_constraints(bbox, (w, h))
        x0, y0, w, h = crop
        img = mx.image.fixed_crop(img, x0, y0, w, h)

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = timage.imresize(img, self._width, self._height, interp=interp)
        bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._anchors is None:
            return img, bbox.astype(img.dtype)
        
        #如果有anchors的输入，则执行下面的运算。计算以前的格式是：
        #gt_bboxes里面是一个图像上的box位置，实际像素点位 当前图片里面的框框个数x4
        #gt_ids里面是对应图像上的label，当前图片里面的框框个数x1
        #下面为batch_size腾出空间
        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
        cls_targets, box_targets, _ = self._target_generator(
            self._anchors, None, gt_bboxes, gt_ids)
        return img, cls_targets[0], box_targets[0]