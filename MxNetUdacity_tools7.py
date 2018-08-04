# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 20:55:38 2018

@author: Bird
"""
from mxnet.gluon import SymbolBlock
import mxnet as mx
from mxnet.gluon import nn

from gluoncv.nn.feature import _parse_network

'''
SymbolBlock
    给网络增加旁路的功能函数。在正式的帮助中有例子。
    注意，增加旁路以后，SymbolBlock会输出一个网络。这个网络的输入，仍然是原来网络的输入。
    而网络的输出是那些在参数outputs中指定了的引出旁路。
    这些层抽出来以后，与原来的网络仍然是联通的。不像使用切片的方法抽出层以后，原来的链接就断开了
    抽出层网络的名称官方的例子提供了一种方法。
    Examples
    --------
    >>> # To extract the feature from fc1 and fc2 layers of AlexNet:
    >>> alexnet = gluon.model_zoo.vision.alexnet(pretrained=True, ctx=mx.cpu(),
                                                 prefix='model_')
    >>> inputs = mx.sym.var('data')
    >>> out = alexnet(inputs)
    >>> internals = out.get_internals()
    >>> print(internals.list_outputs())
    ['data', ..., 'model_dense0_relu_fwd_output', ..., 'model_dense1_relu_fwd_output', ...]
    >>> outputs = [internals['model_dense0_relu_fwd_output'],
                   internals['model_dense1_relu_fwd_output']]
    >>> # Create SymbolBlock that shares parameters with alexnet
    >>> feat_model = gluon.SymbolBlock(outputs, inputs, params=alexnet.collect_params())
    >>> x = mx.nd.random.normal(shape=(16, 3, 224, 224))
    >>> print(feat_model(x))

回到FeatureExpander：
    它不但从网络中可以抽出一些层，而且能够以最后抽出的那个层作为输入，增加一些层。
    这些增加的层是串联的。
    每一个增加的层，都自动制造一个旁路输出。
    Parameters:
        network 要抽取名称的网络
        outputs (str or list of str) – 需要抽出挪作他用的层。
                                        最后一层的输出将会作为增加层的输入。
        num_filters (list of int) – 给这些抽出的层上面增加卷积层，定义卷积的filter个数
        use_1x1_transition (bool) – 这样可是使得卷积的输入和输出个数相同
        use_bn (bool) – 卷积输出完以后就通过一个BatchNorm
        reduce_ratio (float) – 当使用use_1x1_transition标记输入等于输出的时候，可以对
                    卷积的filter个数做有个比例调节。但是不能太小，因为受到min_depth的限制
        min_depth (int) – 限制最少的卷积层filter个数，保安参数。防止设计的过小
        global_pool (bool) – 通过激活函数以后，是否通过一个Pool函数
        pretrained (bool) – 是否使用预训练的权重
        ctx (Context) – 选择使用计算装置的种类

        inputs (list of str) – 这个是核心部位的抽出函数SymbolBlock所需要的东西。
'''



class FeatureExpander(SymbolBlock):

    def __init__(self, network, outputs, num_filters, use_1x1_transition=True,
                 use_bn=True, reduce_ratio=1.0, min_depth=128, global_pool=False,
                 pretrained=False, ctx=mx.cpu(), inputs=('data',)):
        inputs, outputs, params = _parse_network(network, outputs, inputs, pretrained, ctx)
        # append more layers
        #在生成的新网络的最后一个输出增加新的层
        y = outputs[-1]
        #权值初始化装置
        weight_init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2)
        for i, f in enumerate(num_filters):
            if use_1x1_transition:
                num_trans = max(min_depth, int(round(f * reduce_ratio)))
                y = mx.sym.Convolution(
                    y, num_filter=num_trans, kernel=(1, 1), no_bias=use_bn,
                    name='expand_trans_conv{}'.format(i), attr={'__init__': weight_init})
                if use_bn:
                    y = mx.sym.BatchNorm(y, name='expand_trans_bn{}'.format(i))
                y = mx.sym.Activation(y, act_type='relu', name='expand_trans_relu{}'.format(i))
            y = mx.sym.Convolution(
                y, num_filter=f, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
                no_bias=use_bn, name='expand_conv{}'.format(i), attr={'__init__': weight_init})
            if use_bn:
                y = mx.sym.BatchNorm(y, name='expand_bn{}'.format(i))
            y = mx.sym.Activation(y, act_type='relu', name='expand_reu{}'.format(i))
            outputs.append(y)
        if global_pool:
            outputs.append(mx.sym.Pooling(y, pool_type='avg', global_pool=True, kernel=(1, 1)))
        super(FeatureExpander, self).__init__(outputs, inputs, params)

from mxnet import gluon

class FeatureExpander_IsolatedParams(SymbolBlock):
# 可以将基础模型和增加层模型的参数独立开来的FeatureExpander
    def __init__(self, network, outputs, num_filters, use_1x1_transition=True,
                 use_bn=True, reduce_ratio=1.0, min_depth=128, global_pool=False,
                 pretrained=False, ctx=mx.cpu(), inputs=('data',)):
        
        self.IsolatedParams = gluon.ParameterDict()
        
        inputs, outputs, params = _parse_network(network, outputs, inputs, pretrained, ctx)
        # append more layers
        #在生成的新网络的最后一个输出增加新的层
        y = outputs[-1]
        #权值初始化装置
        weight_init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2)
        for i, f in enumerate(num_filters):
            if use_1x1_transition:
                num_trans = max(min_depth, int(round(f * reduce_ratio)))
#                y = mx.sym.Convolution(
#                    y, num_filter=num_trans, kernel=(1, 1), no_bias=use_bn,
#                    name='expand_trans_conv{}'.format(i), attr={'__init__': weight_init})
                
                Conv2D_1 = nn.Conv2D(channels=num_trans,kernel_size=(1, 1),use_bias=not(use_bn),weight_initializer=weight_init,prefix='expand_trans_conv{}_'.format(i))
                y = Conv2D_1(y)
                self.IsolatedParams.update(Conv2D_1.collect_params())
#                name='expand_trans_conv{}'.format(i)
                if use_bn:
                    y = mx.sym.BatchNorm(y, name='expand_trans_bn{}'.format(i))
                y = mx.sym.Activation(y, act_type='relu', name='expand_trans_relu{}'.format(i))
                
#            y = mx.sym.Convolution(
#                y, num_filter=f, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
#                no_bias=use_bn, name='expand_conv{}'.format(i), attr={'__init__': weight_init})
            
            Conv2D_2 = nn.Conv2D(channels=f,kernel_size=(3, 3),padding=(1, 1),strides=(2, 2),use_bias=not(use_bn),weight_initializer=weight_init,prefix='expand_conv{}_'.format(i))
            y = Conv2D_2(y)
            self.IsolatedParams.update(Conv2D_2.collect_params())
            
            if use_bn:
                y = mx.sym.BatchNorm(y, name='expand_bn{}'.format(i))
            y = mx.sym.Activation(y, act_type='relu', name='expand_reu{}'.format(i))
            outputs.append(y)
        if global_pool:
            outputs.append(mx.sym.Pooling(y, pool_type='avg', global_pool=True, kernel=(1, 1)))
        super(FeatureExpander_IsolatedParams, self).__init__(outputs, inputs, params)
