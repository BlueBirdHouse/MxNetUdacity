# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:10:16 2018

@author: Bird
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 08:41:38 2018

@author: Bird
"""

#%% 常数
dataDir = 'F:/Temps/DataSets/ImageNet/'

modelsDir = 'F:/Temps/Models_tmp/'

#%% 
#from gluoncv.model_zoo.ssd.anchor import SSDAnchorGenerator

from mxnet import gluon
from mxnet import nd
import numpy as np

#from gluoncv.utils import viz


#%% 
class SSDAnchorGenerator(gluon.HybridBlock):
    """
    功能非常强的猫框生成函数。给定一个图片，就可以利用这个图片生成猫框。
    生成的猫框，是以[中心坐标,宽度，高度]的方式表示的，单位是实际的像素坐标。
    生成规则是：
    以输入这一层x的每一个像素为步数，以step为步长生成猫框。
    简算方法是：
    内部变量anchors仅仅生成一次，然后针对不同的图片x，仅仅是对内部变量的重采样。
    """
    def __init__(self, index, im_size, sizes, ratios, step, alloc_size=(128, 128),
                 offsets=(0.5, 0.5), clip=False, **kwargs):
        '''
        Parameters
        ----------
        index : int
            用于生成本层名字而用到的编号
        sizes : iterable of floats
            猫框的大小列表
        ratios : iterable of floats
            猫框的形状列表
        step : int or float
            每隔几个像素，设置一个猫框的中心坐标点。
            标准的做法是每一个像素生成一个猫框。但这种方法更灵活。可以不受到图像本身的限制
        alloc_size : tuple of int
            相当于是先虚拟一个较大的图片，这个图片的长宽决定在x和y两个方向上走多少步。
            实际产生的猫框中心位置以这个图片的长宽为步数，以step的设置为步长，生成。
            最后，系统会根据当前的图片输入，裁剪合适的猫框出来的。
            所以只要设置稍大一点的值即可。    
        offsets : tuple of float
            给生成的猫框中心坐标增加一个偏移量
        im_size ：
            最初通入网络图像的大小。
            起到保安作用。由于SSD在处理过程中执行的是像素缩减。所以后来的图像一定不会比最初通入的图像大。
            这样，如果这一层的输入比最初的图像还大，则会根据最初的图像大小做裁剪。
        ''' 
        
        super(SSDAnchorGenerator, self).__init__(**kwargs)
        assert len(im_size) == 2
        self._im_size = im_size
        self._clip = clip
        self._sizes = (sizes[0], np.sqrt(sizes[0] * sizes[1]))
        self._ratios = ratios
        anchors = self._generate_anchors(self._sizes, self._ratios, step, alloc_size, offsets)
        self.anchors = self.params.get_constant('anchor_%d'%(index), anchors)

    def _generate_anchors(self, sizes, ratios, step, alloc_size, offsets):
        """Generate anchors for once."""
        assert len(sizes) == 2, "SSD requires sizes to be (size_min, size_max)"
        anchors = []
        for i in range(alloc_size[0]):
            for j in range(alloc_size[1]):
                cy = (i + offsets[0]) * step
                cx = (j + offsets[1]) * step
                # ratio = ratios[0], size = size_min or sqrt(size_min * size_max)
                r = ratios[0]
                anchors.append([cx, cy, sizes[0], sizes[0]])
                anchors.append([cx, cy, sizes[1], sizes[1]])
                # size = sizes[0], ratio = ...
                for r in ratios[1:]:
                    sr = np.sqrt(r)
                    w = sizes[0] * sr
                    h = sizes[0] / sr
                    anchors.append([cx, cy, w, h])
        return np.array(anchors).reshape(1, 1, alloc_size[0], alloc_size[1], -1)

    @property
    def num_depth(self):
        """Number of anchors at each pixel."""
        return len(self._sizes) + len(self._ratios) - 1

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, anchors):
        '''
        原理是，在初始化的时候生成很多的猫框存储好
        根据不同的输入图片，裁剪猫框的个数。
        这样就不需要为每一个不同尺寸的图片反复生成猫框了。
        节约了运算。
        '''
        a = F.slice_like(anchors, x * 0, axes=(2, 3))
        a = a.reshape((1, -1, 4))
        if self._clip:
            cx, cy, cw, ch = a.split(axis=-1, num_outputs=4)
            H, W = self._im_size
            a = F.concat(*[cx.clip(0, W), cy.clip(0, H), cw.clip(0, W), ch.clip(0, H)], dim=-1)
        return a.reshape((1, -1, 4))


def box_xywh2XYXY(boxes):
    #将中心坐标+长宽 模式表示的猫框转换为对角点坐标的模式
    x = boxes[:,:,0]
    y = boxes[:,:,1]
    w = boxes[:,:,2]
    h = boxes[:,:,3]
    
    X_min = (x - h/2).expand_dims(axis = 2)
    Y_min = (y - w/2).expand_dims(axis = 2)
    X_max = (x + h/2).expand_dims(axis = 2)
    Y_max = (y + w/2).expand_dims(axis = 2)

    return nd.concat(X_min,Y_min,X_max,Y_max,dim=2)


