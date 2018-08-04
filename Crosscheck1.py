# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 09:24:09 2018

@author: Bird
交叉检验1

用来验证GluonCV官方配置训练出来的SSD 

"""
#%%
from MxNetUdacity_tools2 import SSD
from MxNetUdacity_tools6 import NormalizedBoxCenterDecoder,MultiPerClassDecoder

from gluoncv import data
from gluoncv import utils

from mxnet.ndarray import softmax,concat,slice_axis
from mxnet.ndarray.contrib import box_nms

from matplotlib import pyplot as plt
#%% 
classes = data.VOCDetection.CLASSES

name = 'resnet50_v1'
base_size = 512
features=['stage3_activation5', 'stage4_activation2']
filters=[512, 512, 256, 256]
sizes=[51.2, 102.4, 189.4, 276.4, 363.52, 450.6, 492]
ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2
steps=[16, 32, 64, 128, 256, 512]

pretrained=True

net = SSD(network = name, base_size = base_size, features = features, 
          num_filters = filters, sizes = sizes, ratios = ratios, steps = steps,
              pretrained=pretrained, classes=classes)

#net.load_parameters('F:/Temps/Models_tmp/ssd_512_resnet50_v1_voc-9c8b225a.params',allow_missing=False)
#net.initialize()
net.load_parameters('F:/Temps/Models_tmp/UnTrainedGPU/' +'myssd.params')

#%% 
#利用官方VOC数据库验证结果
im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/street_small.jpg?raw=true',
                          path='street_small.jpg')
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)

#利用尤达数据库验证结果
#dataDir = 'F:/Temps/DataSets/object-dataset/'
#x, img = data.transforms.presets.ssd.load_test(dataDir + '1479500069582379555.jpg', short=512)

print('Shape of pre-processed image:', x.shape)

cls_preds, box_preds, anchors = net(x)

#%% 解析结果

num_classes= len(classes) + 1
nms_thresh=0.45
nms_topk=-1 #-1 400
post_nms=100
stds=(0.1, 0.1, 0.2, 0.2)

bbox_decoder = NormalizedBoxCenterDecoder(stds)
cls_decoder = MultiPerClassDecoder(num_classes, thresh=0.01)

bboxes = bbox_decoder(box_preds, anchors)
cls_ids, scores = cls_decoder(softmax(cls_preds, axis=-1))
results = []
for i in range(num_classes - 1):
    cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i+1)
    score = scores.slice_axis(axis=-1, begin=i, end=i+1)
    # per class results
    per_result = concat(*[cls_id, score, bboxes], dim=-1)
    results.append(per_result)
result = concat(*results, dim=1)
if nms_thresh > 0 and nms_thresh < 1:
    result = box_nms(
        result, overlap_thresh=nms_thresh, topk=nms_topk, valid_thresh=0.01,
        id_index=0, score_index=1, coord_start=2, force_suppress=False)
    if post_nms > 0:
        result = result.slice_axis(axis=1, begin=0, end=post_nms)
        
class_IDs = slice_axis(result, axis=2, begin=0, end=1)
scores = slice_axis(result, axis=2, begin=1, end=2)
bounding_boxs = slice_axis(result, axis=2, begin=2, end=6)



#%% 显示输出
ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                         class_IDs[0], class_names=net.classes,thresh=0.5)
plt.show()


