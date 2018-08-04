# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 09:47:23 2018

@author: Bird

GPU和CPU数据分发功能函数
"""
#%%
from mxnet import ndarray
from mxnet.gluon.utils import split_data
from mxnet.ndarray import slice_axis
from mxnet import nd
import mxnet as mx
from mxnet.gluon.utils import split_and_load

#%%
def split_and_load_WithCPU(data, GPU_list, batch_axis=0, CPU_percentage = 0.3):
    """将数据在GPU和CPU之间分发
    """
    size = data.shape[batch_axis]
    CUP_size = round(size * CPU_percentage)
    
    CUP_data = slice_axis(data, axis=batch_axis, begin=0, end=CUP_size) 
    CUP_data = CUP_data.as_in_context(mx.cpu())
    
    GPU_data = slice_axis(data, axis=batch_axis, begin=CUP_size, end= size) 
    GPU_data = split_and_load(GPU_data, GPU_list, batch_axis=batch_axis, even_split=False) 
    
    return[CUP_data] + GPU_data
    
