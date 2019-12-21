'''
@Author: your name
@Date: 2019-12-19 12:11:20
@LastEditTime : 2019-12-20 11:17:30
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /Machine_Learning/Decision Tree/entropy_enhance.py
'''
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

featList = ['年龄','有工作','有自己的房子','信贷情况']


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

    print("最佳划分为第%d个特征，是“%s”,信息增益为%.3f" % (bestFeatVec,featList[bestFeatVec],bestInfoGain))

    return bestFeatVec

# optimalPartition(dataSet)
# print(float(dataSet.shape[0]))

'''
@description: 数据集已经处理了所有属性，但是类标签依然不是唯一的,采用多数判决的方法决定该子节点的分类
        即统计yList中出现次数最多的元素（类标签）
@param {type} 
    yList: 类别标签列
@return: 
    sortedClassCount[0][0]：出现次数最多的元素（类标签）
'''
def majorityCnt(yList):
    yCount = {}
    #统计yList中每个类别出现的次数：
    for num in yList:
        if num not in yCount.keys():
            yCount[num] = 0
        yCount[num] += 1
    sortedyCount = sorted(yCount.items(),key=operator.itemgetter(1),reverse=True)

    return sortedyCount[0][0]
    


'''
@description: 创建决策树
@param {type} 
    dataSet：训练数据集
    featList：分类属性标签
    bestFeatLists：存储选择的最优特征标签
@return: tree 决策树
'''
def createTree(dataSet,featList,bestFeatLists):
    #取训练数据集的最后一列，即分类标签。注意这里是list类型。区分dataSet[:,-1]这是数组形式
    yList = [example[-1] for example in dataSet]

    # 如果类别完全相同，则停止继续划分，
    # 即yList中所有类别都是同一数据值（该类别数值个数等于列表长度）
    if yList.count(yList[0]) == len(yList):
        return yList[0]
    

    # 数据集已经处理了所有属性，但是类标签依然不是唯一的，
    # 则采用多数判决的方法决定该子节点的分类
    # 为什么要如此判断？dataSet的列是不断减少的，dataSet某一行的长度，就是列
    if len(dataSet[0]) == 1:
        return majorityCnt(yList)
    
    #选择最优划分的特征index
    bestFeatVec = optimalPartition(dataSet)
    # 最优特征index所对应的分类标签，作为树的根节点
    bestFeatLabel = featList[bestFeatVec]
    # 存储选择的最优特征标签
    bestFeatLists.append(bestFeatLabel)

    # 将最优划分特征值作为当前（子）树的根节点，生成初始决策树（用字典表示一个树结构）
    myTree = {bestFeatLabel:{}}

    #删除已经选择过的特征
    del(featList[bestFeatVec])
    print('featList',featList)
    
    # 得到训练集中所有最优特征那一列所对应的值
    featValue = [example[bestFeatVec] for example in dataSet]
    # 去掉重复的属性值，得到最优特征下的子类
    categories = set(featValue)

    # 遍历最优特征列所对应的值，创建决策树
    # 如“年龄”是最优特征，则遍历“老”“中”“青”三个子类
    for category in categories:
        # 根据当前数据集、最优划分的特征index以及每个分类（条件）得到（条件下的子集）
        subDataSet = np.array(currentConditionSet(dataSet,bestFeatVec,category))
        # 递归地调用创建决策树的方法，将递归调用的结果作为当前树节点的一个分支
        myTree[bestFeatLabel][category] = createTree(subDataSet,featList,bestFeatLists)

    return myTree

if __name__ == "__main__":
    bestFeatLists = []
    myTree = createTree(dataSet,featList,bestFeatLists)
    print(myTree)
    


