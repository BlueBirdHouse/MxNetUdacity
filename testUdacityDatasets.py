# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 17:05:16 2018

@author: Bird

测试一下尤达数据库中的功能，并将VOC数据库和尤达数据库功能做比较。
如果这个函数在目标机器上输出是一致的，说明目标机器的配置是正常的。

"""
#%% 
from mxnet import gluon
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd

from matplotlib import pyplot as plt

from MxNetUdacity_tools1 import udacityDetection
from MxNetUdacity_tools9 import SSDDefaultTrainTransform

from gluoncv.data import VOCDetection
from gluoncv.utils import viz
#from gluoncv.data.transforms import presets
from gluoncv.data import DetectionDataLoader
#%% 
dataDir = 'F:/Temps/DataSets/object-dataset/'
width = 512
height = 512  

batch_size = 4
maxTrain_inOneEpoch = 10
epoch_num = 2
lambd = 1/6
Models_tmp_Dir = 'F:/Temps/Models_tmp/' 

#使用训练元件
ctx = [mx.cpu()]

#%% 
train_dataset_official = VOCDetection()
train_dataset = udacityDetection(root = dataDir)

#%% 这个值应该是
print('Training images(应该是21601):', len(train_dataset))


train_image, train_label = train_dataset[11960]
'''
应该是：
image: (1200, 1920, 3)
bboxes: (X, 4) class ids: (X, 1)
'''
bboxes = train_label[:, :4]
cids = train_label[:, 4:5]
print('image:', train_image.shape)
print('bboxes:', bboxes.shape, 'class ids:', cids.shape)

ax = viz.plot_bbox(train_image.asnumpy(), bboxes, labels=cids, class_names=train_dataset.classes)
plt.show()

#%% 测试转换函数，尝试将图片缩小到标准图片
train_transform = SSDDefaultTrainTransform(width, height)
train_image2, train_label2 = train_transform(train_image, train_label)
print('tensor shape:', train_image2.shape)
print('label shape:', train_label2.shape)

train_image2 = train_image2.transpose((1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
train_image2 = (train_image2 * 255).clip(0, 255)
ax = viz.plot_bbox(train_image2.asnumpy(), train_label2[:, :4],
                   labels=train_label2[:, 4:5],
                   class_names=train_dataset.classes)
plt.show()

batch_size = 2  # for tutorial, we use smaller batch-size
num_workers = 0  # you can make it larger(if your CPU has more cores) to accelerate data loading

train_loader = DetectionDataLoader(train_dataset.transform(train_transform), batch_size, shuffle=True,
                                   last_batch='rollover', num_workers=num_workers)
for ib, batch in enumerate(train_loader):
    if ib > 3:
        break
    print('data:', batch[0].shape, 'label:', batch[1].shape)

