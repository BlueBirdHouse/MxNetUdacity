# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 09:35:15 2018

@author: Bird
交叉检验3

使用尤达数据库，GPU/CPU 训练SSD。

"""
#%% 
from gluoncv.data import DetectionDataLoader

from MxNetUdacity_tools2 import SSD
from MxNetUdacity_tools5 import SSDMultiBoxLoss
from MxNetUdacity_tools1 import udacityDetection
from MxNetUdacity_tools9 import SSDDefaultTrainTransform

from mxnet import autograd
import mxnet as mx
from mxnet.gluon.utils import split_and_load

#%% 
dataDir = 'F:/Temps/DataSets/object-dataset/'
width = 512
height = 512  

batch_size = 20
maxTrain_inOneEpoch = 3
epoch_num = 1
lambd = 1/4
Models_tmp_Dir = 'F:/Temps/Models_tmp/' 

#使用训练元件
#ctx = [mx.gpu(i) for i in range(mx.context.num_gpus())]
ctx = [mx.cpu()]

#%% 网络
train_dataset = udacityDetection(root = dataDir)

classes = train_dataset.classes

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
              pretrained=pretrained, classes=classes,ctx=ctx[0])
net.initialize(verbose = True,ctx = ctx)
#net.load_parameters(Models_tmp_Dir + 'ssd_512_resnet50_v1_voc-9c8b225a.params')
net.hybridize(active= True,static_alloc=True,static_shape = True)

#%% 生成DataLoader
x = mx.nd.zeros(shape=(1, 3, base_size, base_size),ctx=ctx[0])
cls_preds, box_preds, anchors = net(x)
anchors = anchors.as_in_context(mx.cpu())
train_transform = SSDDefaultTrainTransform(width, height,anchors)
#通过train_dataset.transform(train_transform)转换以后是LazyTransformDataset
#这是Dataset基类提供方法导致的，这个对象本身没有什么作用，所以要成一行
train_loader = DetectionDataLoader(train_dataset.transform(train_transform), batch_size, shuffle=True,
                                   last_batch='rollover', num_workers=0)

#%% 生成训练目标并评估
mbox_loss = SSDMultiBoxLoss(lambd=lambd)

all_params = net.collect_params()
trainer = mx.gluon.Trainer(all_params,'sgd')


#%% 训练

for epoch in range(epoch_num): 
    
    for ib, batch in enumerate(train_loader):
        if ib > maxTrain_inOneEpoch:
            break
        print('data:', batch[0].shape)
        print('class targets:', batch[1].shape)
        print('box targets:', batch[2].shape)
        
        data = split_and_load(batch[0], batch_axis=0,ctx_list=ctx)
        cls_targets = split_and_load(batch[1], batch_axis=0,ctx_list=ctx)
        box_targets = split_and_load(batch[2], batch_axis=0,ctx_list=ctx)
        
        with autograd.record():
            #可以将数据一个一个的通入，以减少一次性的内存消耗（参考train_ssd.py）
            cls_preds = []
            box_preds = []
            for x in data:
                cls_pred, box_pred, _ = net(x)
                #这两句如果能去掉则去掉
                cls_preds.append(cls_pred)
                box_preds.append(box_pred)
                
            sum_loss, cls_loss, box_loss = mbox_loss(cls_preds, box_preds, cls_targets, box_targets)
        autograd.backward(sum_loss)
        #一般情况下应该等于batch_size，但是在这里loss已经被一般化了
        trainer.step(1)

        cls_loss = cls_loss[0].as_in_context(mx.cpu())
        box_loss = box_loss[0].as_in_context(mx.cpu())
        clsloss_Num = lambd * (cls_loss.asnumpy()).mean()
        boxloss_Num = (1 - lambd) * (box_loss.asnumpy()).mean()
        print('[Epoch {}][Batch {}], {}={:.3f}, {}={:.3f}'.format(
                    epoch, ib , 'CLS Loss', clsloss_Num, 'Box Loss', boxloss_Num))
        print('___________________')
  
    
#%% 保存参数
    
net.save_parameters(Models_tmp_Dir +'myssd.params')    




