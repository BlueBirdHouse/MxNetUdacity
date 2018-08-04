# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:59:24 2018

@author: Bird

生成udacityDetection数据集，这个数据集合是模拟官方的VOCDetection写的
由于udacity数据集使用了csv文件直接存储数据信息并且使用大小不变的图像
所以不使用label缓冲的策略
"""
from gluoncv.data.base import VisionDataset

from mxnet.image import imread
from mxnet import nd

import os
import numpy as np

from MxNetUdacity_tools import explainFlabels,explainFlabels_crowdai


class udacityDetection(VisionDataset):
    CLASSES = ('car', 'pedestrian', 'truck', 'trafficLight', 'biker')

    def __init__(self, root,
             transform=None, index_map=None,
             splits = ('labels','labels_crowdai_clean'),
             imShape = (1200,1920)
             ):
        super(udacityDetection, self).__init__(root)
        
        #数据库的根目录
        self._root = os.path.expanduser(root)
        #存储图像预处理方法的地方
        self._transform = transform
        #存储有关数据库本身的信息
        self._splits = splits
        #存储图片大小,这个数据库的图片大小是固定的
        self._im_shape = imShape
        
        
        #这个里面存储路径和对应的label集合
        self._items = self._load_items(self._splits)
        #一个文件名的列表，用来保证读入数据是有序的
        self._fileNames = sorted(self._items.keys())
        
        #一个分类名称和分类标记的对照表
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        
        #缓冲区
        #仅仅缓冲一个图片内容+信息的缓冲区
        self._imgInfo = {'fileName':0,'img':nd.array([0]),'shape':(0,0,0)}
        #将缓冲label功能改为对整个label做一个检查
        [self._load_label(idx) for idx in range(len(self))]
    
    def _load_items(self, splits):
        #读入数据库内容描述文件
        dick = {}
        if splits[0] == 'labels':
            dick = explainFlabels(dick,self._root)
        if splits[1] == 'labels_crowdai_clean':
            dick = explainFlabels_crowdai(dick,self._root)
        return dick
    
    def __str__(self):
        #将数据库信息打印为容易识别的方式
        detail = '+'.join(self._splits)
        return self.__class__.__name__ + '(' + detail + ')'
    
    @property
    def classes(self):
        #输出分类名称
        return type(self).CLASSES
    
    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        #标签有效性检查
        assert xmin >= 0 and xmin < width, (
            "xmin must in [0, {}), given {}".format(width, xmin))
        assert ymin >= 0 and ymin < height, (
            "ymin must in [0, {}), given {}".format(height, ymin))
        assert xmax > xmin and xmax <= width, (
            "xmax must in (xmin, {}], given {}".format(width, xmax))
        assert ymax > ymin and ymax <= height, (
            "ymax must in (ymin, {}], given {}".format(height, ymax))
    
    def __len__(self):
        #输出数据库的大小
        return len(self._items)
    
    
    def _image_Info(self,fileName):
        #读取图像，并给出图像的大小
        #然后将图像缓冲，下一次调用这个图像的时候，如果在缓冲，就直接调用，如果不在就重新读取
        if self._imgInfo['fileName'] == fileName:
            img = self._imgInfo['img']
            imShape = self._imgInfo['shape']
            return img,imShape
        
        img = imread(self._root + fileName, 1)
        self._imgInfo['fileName'] = fileName
        self._imgInfo['img'] = img
        self._imgInfo['shape'] = img.shape
        return img, img.shape
        
    
    def _load_label(self, idx):
        #给出idx所对应的label值并完成缓冲操作，以及label的正常性检查
        afileName = self._fileNames[idx]

        width = self._im_shape[1]
        height = self._im_shape[0]
            
        label = []
        labelList = self._items[afileName]
        for alabel in labelList:
            
            cls_id = alabel[4]
            
            xmin = alabel[0]
            ymin = alabel[1]
            xmax = alabel[2]
            ymax = alabel[3]
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError:
                raise RuntimeError("Invalid label at {}, {}")
            label.append([xmin, ymin, xmax, ymax, cls_id])
        return np.array(label)

    
    def __getitem__(self, idx):
        #输出图像和对应的标签
        afileName = self._fileNames[idx]
        
        img, _ = self._image_Info(afileName)
        
        label = self._load_label(idx)
        
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

