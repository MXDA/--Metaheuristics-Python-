import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import math
import copy
import os

'''
构造VRPTW初始解
输入cusnum：      顾客数目
输入a：           顾客左时间窗
输入demands：     顾客需求量
输入cap：         车辆最大装载量
输出init_vc：     初始解
'''


def init(cusnum, a, demands, cap):
    j = np.ceil(np.random.rand() * cusnum).astype(int)  # 从所有顾客中随机选择一个顾客
    k = 1  # 使用车辆数目，初始设置为1
    init_vc = np.empty(cusnum, dtype=object)
    # 按照如下序列，遍历每个顾客，并执行以下步骤
    if j == 1:
        seq = np.array([x + 1 for x in range(cusnum)])
    elif j == cusnum:
        seq = np.array([x + 1 for x in range(j - 1)])
        seq = np.insert(seq, 0, cusnum)
    else:
        seq1 = np.array([x + 1 for x in range(j - 1)])
        seq2 = np.array([x + 1 for x in range(j - 1, cusnum)])
        seq = np.append(seq2, seq1)
    # 开始遍历
    route = np.array([])  # 存储每条路径上的顾客
    load = 0  # 初始路径上在仓库的装载量为0
    i = 0
    while i < cusnum:
        # 如果没有超过容量约束，则按照左时间窗大小，将顾客添加到当前路径
        if load + demands[seq[i] - 1] <= cap:
            load = load + demands[seq[i] - 1]  # 初始在仓库的装载量增加
            # 如果当前路径为空，直接将顾客添加到路径中
            if np.size(route) == 0:
                route = np.append(route, seq[i])
            # 如果当前路径只有一个顾客，再添加新顾客时，需要根据左时间窗大小进行添加
            elif np.size(route) == 1:
                if a[seq[i] - 1] <= a[route[0] - 1]:
                    route = np.insert(route, 0, seq[i])
                else:
                    route = np.append(route, seq[i])
            else:
                lr = np.size(route)  # 当前路径长度,则有lr-1对连续的顾客
                flag = 0  # 标记是否存在这样1对顾客，能让seq(i)插入两者之间
                if a[seq[i] - 1] < a[route[0] - 1]:
                    route = np.insert(route, 0, seq[i])
                elif a[seq[i] - 1] > a[route[-1] - 1]:
                    route = np.append(route, seq[i])
                else:
                    # 遍历这lr-1对连续的顾客的中间插入位置
                    for m in range(lr - 1):
                        if (a[seq[i] - 1] >= seq[route[m] - 1]) and (a[seq[i] - 1] <= a[route[m + 1] - 1]):
                            route = np.insert(route, m + 1, seq[i])
                            break
            if i == cusnum - 1:
                init_vc[k - 1] = route
                break
            i = i + 1
        else:  # 一旦超过车辆装载量约束，则需要增加一辆车
            # 先储存上一辆车所经过的顾客
            init_vc[k - 1] = route
            # 然后将route清空，load清零,k加1
            route = np.array([])
            load = 0
            k = k + 1

    init_vc = deal_vc(init_vc)
    return init_vc


'''
将VC中空的配送路线删除
输入VC：          配送方案，即每辆车所经过的顾客
输出FVC：         删除空配送路线后的VC
'''


def deal_vc(VC):
    index = np.array([]).astype(int)
    for i in range(np.size(VC)):
        if VC[i] is None or np.size(VC[i]) == 0:
            index = np.append(index, i)
    if np.size(index) != 0:
        VC = np.delete(VC, index)  # 删除cell数组中的空元胞
    return VC


'''
配送方案与个体之间进行转换
输入VC：          配送方案
输入N：           染色体长度
输入cusnum：      顾客数目
输出individual：  由配送方案转换成的个体
'''


def change(VC, N, cusnum):
    NV = np.size(VC)  # 车辆使用数目
    individual = np.array([]).astype(int)
    for i in range(NV):
        if cusnum + i + 1 <= N:
            individual = np.append(individual, VC[i])
            individual = np.append(individual, cusnum + i + 1)
        else:
            individual = np.append(individual, VC[i])

    if np.size(individual) < N:  # 如果染色体长度小于N，则需要向染色体添加配送中心编号
        supply = np.array([x + 1 for x in range(cusnum + NV, N)])
        individual = np.append(individual, supply)
    return individual


'''
初始化种群
输入NIND：                种群数目
输入N：                   染色体长度
输入cusnum：              顾客数目
输入init_vc：             初始配送方案
输出Chrom：               初始种群
'''


def init_pop(NIND, N, cusnum, init_vc):
    Chrom = np.zeros((NIND, N)).astype(int)  # 用于存储种群
    individual = change(init_vc, N, cusnum)
    for j in range(NIND):
        Chrom[j, :] = copy.deepcopy(individual)
    return Chrom

