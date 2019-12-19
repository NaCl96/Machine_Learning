'''
@Author: NaCl
@Date: 2019-12-18 20:54:27
@LastEditTime : 2019-12-18 20:54:27
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /Machine_Learning/Decision Tree/entropy_enhance.py
'''

"""
信息增益最优划分
"""
#-------------------------------------------------------------------#

import numpy as np 
from collections import Counter
from math import log
#数据集
dataSet=np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 1],
                  [0, 1, 1, 0, 1],
                  [0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [1, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1],
                  [1, 0, 1, 2, 1],
                  [1, 0, 1, 2, 1],
                  [2, 0, 1, 2, 1],
                  [2, 0, 1, 1, 1],
                  [2, 1, 0, 1, 1],
                  [2, 1, 0, 2, 1],
                  [2, 0, 0, 0, 0]])

X = dataSet[:,4]
y = dataSet[:,-1:]

strs = ['年龄','有工作','有自己的房子','信贷情况','是否申请贷款']


'''
@description: 计算经验熵
@param : 
    dataset:样本数据集合D
@return: 经验熵
'''
def calEntropy(dataSet):
    #返回数据集行数
    numData = len(dataSet)
    #设置字典，保存每个标签出现的次数
    labelCount = {}

    #对每组特征向量进行统计
    for featVec in dataSet:
        #提取标签信息
        label = featVec[-1]
        if label not in labelCount.keys():
            labelCount[label] = 0
        labelCount[label] += 1
    
    entroy = 0.0
    #计算经验熵
    for key in labelCount:
        prob = float(labelCount[key])/numData
        entroy -= prob * log(prob,2)
    
    return entroy


'''
@description: 得到当前特征条件下的小类的所有样本集合（即不包含当前特征的特征样本集）
@param {type}
    dataSet：样本数据集
    curtFeatIndex : 当前用来划分数据集的特征A的位置
    categories：特征A所有可能分类的集合
@return: 
    otherFeatSets：不包含当前特征的特征样本集
'''
def currentConditionSet(dataSet,curtFeatIndex,category):
    otherFeatSets = []

    for featVec in dataSet:
        if featVec[curtFeatIndex] == category:
            otherFeatSet = np.append(featVec[:curtFeatIndex],featVec[curtFeatIndex+1:])
            otherFeatSets.append(otherFeatSet)

    return otherFeatSets


'''
@description: 在选择当前特征的条件下，计算熵，即条件熵
@param {type} 
    dataSet：样本数据集D
    curtFeatIndex：当前用来划分数据集的特征A的位置
    categories：特征A所有可能分类的集合
@return: 条件熵
'''
def calConditionalEnt(dataSet,curtFeatIndex,categories):
    #条件熵初始
    conditionalEnt = 0.0

    # 对于每一个分类，计算选择当前特征的条件下条件熵
    # 比如在选择“年龄”这一特征下，共有“老中青”三个小分类
    for category in categories:
        
        cdtSetCategory = currentConditionSet(dataSet,curtFeatIndex,category)

        prob = len(cdtSetCategory) / float(dataSet.shape[0])

        conditionalEnt += prob * calEntropy(cdtSetCategory)
    
    return conditionalEnt


'''
@description: 计算信息增益
@param {type} 
    baseEntropy：划分样本集合D的熵是为H(D)，即基本熵
    dataSet：样本数据集D
    curtFeatIndex：当前用来划分数据集的特征A的位置
@return: 信息增益
'''

def calInfoGain(baseEntropy,dataSet,curtFeatIndex):

    conditionalEnt = 0.0
    #找出当前特征下的每一个取值，并去除重复
    # 相当于该特征一共有几种分类，如“年龄”这一特征，分为“老中青”三类
    categories = set(dataSet[:,curtFeatIndex])

    #计算划分后的数据子集的条件熵
    conditionalEnt = calConditionalEnt(dataSet,curtFeatIndex,categories)

    # 计算信息增益：g(D,A)=H(D)−H(D|A)
    infoGain = baseEntropy - conditionalEnt
    
    #打印每个特征的信息增益
    print("第%d个特征的增益为%.3f"%(curtFeatIndex,infoGain))

    return infoGain


'''
@description: 寻找最优划分
@param {type} 
        dataSet：数据集
@return: 
'''
def optimalPartition(dataSet):

    bestInfoGain = -1 
    bestFeatVec = -1
    #划分样本集D的熵H(D),即基本熵
    baseEntropy = calEntropy(dataSet)

    for featVec in range(dataSet.shape[1] - 1):
        
        #计算信息增益
        infoGain = calInfoGain(baseEntropy,dataSet,featVec)

        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeatVec = featVec

    print("最佳划分为第%d个特征，是“%s”,信息增益为%.3f" % (bestFeatVec,strs[bestFeatVec],bestInfoGain))

    return bestFeatVec

# optimalPartition(dataSet)
print(float(dataSet.shape[0]))
