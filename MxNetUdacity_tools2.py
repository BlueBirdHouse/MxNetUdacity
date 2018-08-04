# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:53:28 2018

@author: Bird

这个网络直接从官方模型‘ssd_512_resnet50_V1_voc’变换而来。
主要目的是为了研究官方模型的生成原理，并最大限度的采用其默认值。
变换的过程中，省略了检测代码，以及官方函数中为了扩大函数本身试用范围而设计的转换代码。
从而尽可能的保留网络本身的骨干结构，让网络变得清晰并易于学习。 
"""

#%% 包导入区
from mxnet.gluon import HybridBlock
import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon

#from gluoncv.nn.feature import FeatureExpander
from MxNetUdacity_tools7 import FeatureExpander_IsolatedParams
from MxNetUdacity_tools7 import FeatureExpander
#from gluoncv.model_zoo.ssd.anchor import SSDAnchorGenerator
from MxNetUdacity_tools3 import SSDAnchorGenerator
#ConvPredictor没什么特别的，就是一个被定义了核大小的卷积层。
from gluoncv.nn.predictor import ConvPredictor


#%% 模型生成区
class SSD(HybridBlock):
    def __init__(self, network, base_size, features, num_filters, sizes, ratios,
                 steps, classes, use_1x1_transition=True, use_bn=True,
                 reduce_ratio=1.0, min_depth=128, global_pool=False, pretrained=False,
                 iou_thresh=0.5, neg_thresh=0.5, negative_mining_ratio=3,
                 stds=(0.1, 0.1, 0.2, 0.2), nms_thresh=0.45, nms_topk=400,
                 anchor_alloc_size=128,ctx=mx.cpu(), **kwargs):
        '''
        参数表：
        network： 见FeatureExpander：network 要抽取名称的网络
        base_size： SSD网络输入的图片大小
        features： 见FeatureExpander：outputs (str or list of str) – 需要抽出挪作他用的层。
        num_filters： 见FeatureExpander：num_filters (list of int) – 给这些抽出的层
            上面增加卷积层，定义卷积的filter个数
        sizes: 见SSDAnchorGenerator：sizes : iterable of floats猫框的大小列表
        ratios： 见SSDAnchorGenerator：ratios： 猫框的形状列表
        steps：见SSDAnchorGenerator：step : int or float
            每隔几个像素，设置一个猫框的中心坐标点。
        classes：除了背景以外的分类名称，其实仅仅使用到有多少个类别
        pretrained：基础模型是否使用预训练模型
        '''
        super(SSD, self).__init__(**kwargs)

        num_layers = len(features) + len(num_filters) + int(global_pool)
        #len(sizes) == num_layers + 1
        sizes = list(zip(sizes[:-1], sizes[1:]))
        
        # num_layers == len(sizes) == len(ratios)

        # num_layers > 0, "SSD require at least one layer, suggest multiple."
        self._num_layers = num_layers
        self.classes = classes
        self.num_classes = len(classes) + 1

        with self.name_scope():
            #从网络中抽出一些层，并增加一些串联的层。
            #每一个串联的层的输出都作为旁路
            #这些旁路都存储在features当中
            self.features = FeatureExpander(
                network=network, outputs=features, num_filters=num_filters,
                use_1x1_transition=use_1x1_transition,
                use_bn=use_bn, reduce_ratio=reduce_ratio, min_depth=min_depth,
                global_pool=global_pool, pretrained=pretrained,ctx=ctx)
            
            #SSD三大功能模块的容器
            self.class_predictors = nn.HybridSequential()
            self.box_predictors = nn.HybridSequential()
            self.anchor_generators = nn.HybridSequential()
            
            #为三大模块做默认配置并装入容器
            asz = anchor_alloc_size
            im_size = (base_size, base_size)
            for i, s, r, st in zip(range(num_layers), sizes, ratios, steps):
                anchor_generator = SSDAnchorGenerator(i, im_size, s, r, st, (asz, asz))
                self.anchor_generators.add(anchor_generator)
                asz = max(asz // 2, 16)  # pre-compute larger than 16x16 anchor map
                num_anchors = anchor_generator.num_depth
                #每一个猫框都要给出对每一个类别的评估
                self.class_predictors.add(ConvPredictor(num_anchors * self.num_classes))
                #每一个猫框都要给出4个位置误差
                self.box_predictors.add(ConvPredictor(num_anchors * 4))

            ##我自己加入初始化过程
#            self.class_predictors.initialize()
#            self.box_predictors.initialize()
#            self.anchor_generators.initialize()
            
#            self.IsolatedParams = gluon.ParameterDict()
#            self.IsolatedParams.update(self.features.IsolatedParams)
#            self.IsolatedParams.update(self.class_predictors.collect_params())
#            self.IsolatedParams.update(self.box_predictors.collect_params())
            
    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        '''
        网络的原理是这样的：
        首先从resnet50_v1这个网络中引出层‘resnetv10_stage3_activation5’和‘resnetv10_stage4_activation2’
        其中，‘resnetv10_stage3_activation5‘的输出直接通过各种分类器得到输出1
        ‘resnetv10_stage4_activation2’的输出也通过各种分类器得到输出2
        但是同时，‘resnetv10_stage4_activation2’输出通过conv->bathNorm->act->conv->bathNorm->act->通过各种分类器得到输出3
        上面的那个没有通过分类器以前的输出再通过一次conv->bathNorm->act->conv->bathNorm->act->通过各种分类器得到输出4
        上面的那个没有通过分类器以前的输出再通过一次conv->bathNorm->act->conv->bathNorm->act->通过各种分类器得到输出5
        上面的那个没有通过分类器以前的输出再通过一次conv->bathNorm->act->conv->bathNorm->act->通过各种分类器得到输出6
        一共有6个输出。
        '''
        features = self.features(x)

        cls_preds = [F.flatten(F.transpose(cp(feat), (0, 2, 3, 1)))
                     for feat, cp in zip(features, self.class_predictors)]
        box_preds = [F.flatten(F.transpose(bp(feat), (0, 2, 3, 1)))
                     for feat, bp in zip(features, self.box_predictors)]
        anchors = [F.reshape(ag(feat), shape=(1, -1))
                   for feat, ag in zip(features, self.anchor_generators)]
        cls_preds = F.concat(*cls_preds, dim=1).reshape((0, -1, self.num_classes))
        box_preds = F.concat(*box_preds, dim=1).reshape((0, -1, 4))
        anchors = F.concat(*anchors, dim=1).reshape((1, -1, 4))
        return [cls_preds, box_preds, anchors]

