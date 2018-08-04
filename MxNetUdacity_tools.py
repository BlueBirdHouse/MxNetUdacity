# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 16:55:12 2018

@author: Bird
"""
#%% 头文件导入段
import csv

#%% 公用常数段
dataDir = 'F:/Temps/DataSets/object-dataset/'

object_dic = {'"car"':0,
              'Car':0,
              '"pedestrian"':1,
              'Pedestrian':1,
              '"truck"':2,
              'Truck':2,
              '"trafficLight"':3,
              '"biker"':4
              }

#%% 函数定义段

def checkValidCoordinate(aCoordinate):
    return (aCoordinate[2] > aCoordinate[0]) and (aCoordinate[3] > aCoordinate[1])


def explainFlabels(dick,dataDir):
    #这个函数解析文件名为labels的txt文件
    
    acsv = csv.reader(open(dataDir + 'labels.csv','r'))
    
    label_list = []
    for arow in acsv:
        arow = arow[0].split(' ')
        
        #不使用那些被挡住的物体
        if arow[5] == '1':
            continue
        
        #类别处理段
        try:
            imagelabel = []
            arow[6] = object_dic[arow[6]]
            imagelabel.append(arow[6])
        except:
            print(arow)
            raise NameError('出现了新的类别。')
                
        #坐标处理段
        aCoordinate = [float(arow[i+1]) for i in range(4)]
    
        #检查框有效性，无效的干脆不要加进来
        if checkValidCoordinate(aCoordinate) == False:
            continue
        
        
        imagelabel =  aCoordinate + imagelabel
        #图像路径处理段
        imgpath = arow[0]
        
        #组合输出结构
        aItem = imagelabel + [imgpath]
        
        label_list.append(aItem)
    
    #下面合并那些针对同一图片的标签
    for aItem in label_list:
        alabel = aItem[0:5]
        afilename = aItem[5]
        try:
            dick[afilename].append(alabel)
        except:
            dick[afilename] = [alabel]
            
    return dick


def explainFlabels_crowdai(dick,dataDir):
    csv1 = csv.reader(open(dataDir + 'labels_crowdai_clean.csv','r'))
    label_list = []
    
    for arow in csv1:
        try:
            imagelabel = []
            arow[5] = object_dic[arow[5]]
            imagelabel.append(arow[5])
        except:
            print(arow)
            raise NameError('出现了新的类别。')
            
        #坐标处理段
        aCoordinate = [float(arow[i]) for i in range(4)]
        #检查框有效性，无效的干脆不要加进来
        if checkValidCoordinate(aCoordinate) == False:
            continue
        
        imagelabel =  aCoordinate + imagelabel
        
        #图像路径处理段
        imgpath = arow[4]
    
        #组合输出结构
        aItem = imagelabel + [imgpath]
    
        label_list.append(aItem)
        
    #下面合并那些针对同一图片的标签
    for aItem in label_list:
        alabel = aItem[0:5]
        afilename = aItem[5]
        try:
            dick[afilename].append(alabel)
        except:
            dick[afilename] = [alabel]
            
    return dick        

#%% 测试代码
'''
dick = {}
dick = explainFlabels(dick)
dick = explainFlabels_crowdai(dick)
'''


