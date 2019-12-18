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
@param {type} 
@return: 
'''
def calEntropy(dataSet):
     
    