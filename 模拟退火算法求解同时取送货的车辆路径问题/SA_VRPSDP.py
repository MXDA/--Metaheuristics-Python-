import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import math
import copy
import os

'''
将当前解转换为配送方案
输入Scurr：       当前解
输入v_num：       车辆最大允许使用数目
输入cusnum：      顾客数目
输入cap：         货车最大装载量
输入demands：     顾客需求量
输入pdemands：    顾客回收量
输入dist：        距离矩阵
输出VC：          配送方案，即每辆车所经过的顾客
输出NV：          车辆使用数目
输出TD：          车辆行驶总距离
输出violate_num： 违反约束路径数目
输出violate_cus： 违反约束顾客数目
'''


def decode(Scurr, v_num, cusnum, cap, demands, pdemands, dist):
    violate_num = 0  # 违反约束路径数目
    violate_cus = 0  # 违反约束顾客数目
    VC = np.empty(v_num, dtype=object)  # 每辆车所经过的顾客
    count = 1  # 车辆计数器，表示当前车辆使用数目
    location0 = np.argwhere(Scurr > cusnum).ravel()  # 找出个体中配送中心的位置
    for i in range(np.size(location0)):
        if i == 0:
            route = copy.deepcopy(Scurr[0: location0[i] + 1])
            route = np.delete(route, np.argwhere(route == Scurr[location0[i]]).ravel())
        else:
            route = copy.deepcopy(Scurr[location0[i - 1]: location0[i] + 1])  # 提取两个配送中心之间的路径
            route = np.delete(route, np.argwhere(route == Scurr[location0[i - 1]]).ravel())  # 删除路径中配送中心序号
            route = np.delete(route, np.argwhere(route == Scurr[location0[i]]).ravel())  # 删除路径中配送中心序号
        VC[count - 1] = route  # 更新配送方案
        count = count + 1  # 车辆使用数目
    if np.size(location0) != 0:
        route = copy.deepcopy(Scurr[location0[-1]:])  # 最后一条路径
        route = np.delete(route, np.argwhere(route == Scurr[location0[-1]]))  # 删除路径中配送中心序号
        VC[count - 1] = route  # 更新配送方案
    VC, NV = deal_vehicles_customer(VC)  # 将VC中空的数组移除
    for j in range(NV):
        route = VC[j]
        flag = JudgeRoute(route, demands, pdemands, cap)  # 判断一条路线是否满足装载量约束，1表示满足，0表示不满足
        if flag == 0:
            violate_cus = violate_cus + np.size(route)  # 如果这条路径不满足约束，则违反约束顾客数目加该条路径顾客数目
            violate_num = violate_num + 1  # 如果这条路径不满足约束，则违反约束路径数目加1
    TD = travel_distance(VC, dist)[0]  # 该方案车辆行驶总距离
    return VC, NV, TD, violate_num, violate_cus


'''
根据VC整理出FVC，将VC中空的配送路线删除
输入VC：          配送方案，即每辆车所经过的顾客
输出FVC：         删除空配送路线后的VC
输出NV：          车辆使用数目
'''


def deal_vehicles_customer(VC):
    index = np.array([]).astype(int)
    for i in range(np.size(VC)):
        if VC[i] is None or np.size(VC[i]) == 0:
            index = np.append(index, i)
    if np.size(index) != 0:
        VC = np.delete(VC, index)  # 删除cell数组中的空元胞
    m = np.size(VC, 0)  # 新方案中所需旅行商的数目
    return VC, m


'''
判断一条配送路线上的各个点是否都满足装载量约束，1表示满足，0表示不满足
输入route：       一条配送路线
输入demands：     顾客需求量
输入pdemands：    顾客回收量
输入cap：         车辆最大装载量
输出flagR：       标记一条配送路线是否满足装载量约束，1表示满足，0表示不满足
'''


def JudgeRoute(route, demands, pdemands, cap):
    flagR = 0  # 初始不满足装载量约束
    Ld, Lc = leave_load(route, demands, pdemands)  # 计算该条路径上离开配送中心和各个顾客时的装载量
    overload_flag = np.argwhere(Lc > cap).ravel()  # 查询是否存在车辆在离开某个顾客时违反装载量约束
    # 如果每个点都满足装载量约束，则将flagR赋值为1
    if (Ld <= cap) and (np.size(overload_flag) == 0):
        flagR = 1
    return flagR


'''
计算某一条路径上离开配送中心和各个顾客时的装载量
输入route：       一条配送路线
输入demands：     顾客需求量
输入pdemands：    顾客回收量
输出Ld：          货车离开配送中心时的装载量
输出Lc：          货车离开各个顾客时的装载量
'''


def leave_load(route, demands, pdemands):
    n = np.size(route)  # 配送路线经过顾客的总数量
    Ld = 0  # 初始车辆在配送中心时的装货量为0
    Lc = np.zeros(n)  # 表示车辆离开顾客时的装载量
    if n != 0:
        for i in range(n):
            if route[i] != 0:
                Ld = Ld + demands[route[i] - 1]
        Lc[0] = Ld + (pdemands[route[0] - 1] - demands[route[0] - 1])
        if n >= 2:
            for j in range(1, n):
                Lc[j] = Lc[j - 1] + (pdemands[route[j] - 1] - demands[route[j] - 1])
    return Ld, Lc


'''
计算当前配送方案违反装载量约束之和
输入VC：          配送方案，即每辆车所经过的顾客
输入demands：     顾客需求量
输入pdemands：    顾客回收量
输入cap：         车辆最大装载量
输出q：           各条配送路线违反装载量约束之和
'''


def violateLoad(VC, demands, pdemands, cap):
    NV = np.size(VC, 0)  # 所用车辆数目
    q = 0
    for i in range(NV):
        route = VC[i]
        n = np.size(route)
        Ld, Lc = leave_load(route, demands, pdemands)
        if Ld > cap:
            q = q + Ld - cap
        for j in range(n):
            if Lc[j] > cap:
                q = q + Lc[j] - cap
    return q


'''
计算一条配送路线的路径长度
输入route：       一条配送路线
输入dist：        距离矩阵
输出p_l：         当前配送路线长度
'''


def part_length(route, dist):
    n = np.size(route)
    p_l = 0
    if n != 0:
        for i in range(n):
            if i == 0:
                p_l = p_l + dist[0, route[i]]
            else:
                p_l = p_l + dist[route[i - 1], route[i]]
    p_l = p_l + dist[route[-1], 0]
    return p_l


'''
计算每辆车所行驶的距离，以及所有车行驶的总距离
输入VC：                  配送方案
输入dist：                距离矩阵
输出sumTD：               车辆行驶总距离
输出everyTD：             每辆车所行驶的距离
'''


def travel_distance(VC, dist):
    n = np.size(VC, 0)  # 车辆数
    everyTD = np.zeros((n, 1))
    for i in range(n):
        part_seq = VC[i]  # 每辆车所经过的顾客
        # 如果车辆不经过顾客，则该车辆所行使的距离为0
        if np.size(part_seq) != 0:
            everyTD[i] = part_length(part_seq, dist)
    sumTD = np.sum(everyTD)  # 所有车行驶的总距离
    return sumTD, everyTD


'''
计算当前解的成本函数
输入VC：          配送方案，即每辆车所经过的顾客
输入dist：        距离矩阵
输入demands：     顾客需求量
输入pdemands：    顾客回收量
输入cap：         车辆最大装载量
输入belta：       违反的装载量约束的惩罚系数
输出cost：        当前配送方案的总成本 f=TD+belta*q
'''


def costFuction(VC, dist, demands, pdemands, cap, belta):
    TD = travel_distance(VC, dist)[0]
    q = violateLoad(VC, demands, pdemands, cap)
    cost = TD + belta * q
    return cost


'''
交换操作
假设当前解为123456，首先随机选择两个位置，然后将这两个位置上的元素进行交换。
比如说，交换2和5两个位置上的元素，则交换后的解为153426。
输入Scurr：       当前解
输出Snew：        经过交换操作后得到的新解
'''


def Swap(Scurr):
    n = np.size(Scurr)
    if n <= 1:
        return Scurr
    seq = np.random.permutation(n)
    I = seq[0: 2]
    i1 = I[0]
    i2 = I[1]
    Snew = copy.deepcopy(Scurr)
    Snew[i1], Snew[i2] = Scurr[i2], Scurr[i1]
    return Snew


'''
逆转操作
假设当前解为123456，首先随机选择两个位置，然后将这两个位置之间的元素进行逆序排列。
比如说，逆转2和5之间的所有元素，则逆转后的解为154326。
输入Scurr：       当前解
输出Snew：        经过逆转操作后得到的新解
'''


def Reversion(Scurr):
    n = np.size(Scurr)
    if n <= 1:
        return Scurr
    seq = np.random.permutation(n)
    I = seq[0: 2]
    i1 = max(I)
    i2 = min(I)
    Snew = copy.deepcopy(Scurr)
    Snew[i1: i2 + 1] = np.flipud(Scurr[i1: i2 + 1])
    return Snew


'''
插入操作
假设当前解为123456，首先随机选择两个位置，然后将这第一个位置上的元素插入到第二个元素后面。
比如说，第一个选择5这个位置，第二个选择2这个位置，则插入后的解为125346。
输入Scurr：       当前解
输出Snew：        经过插入操作后得到的新解
'''


def Insertion(Scurr):
    n = np.size(Scurr)
    if n <= 1:
        return Scurr
    seq = np.random.permutation(n)
    I = seq[0: 2]
    i1 = I[0]
    i2 = I[1]
    if i1 < i2:
        Snew = copy.deepcopy(Scurr[0: i1])
        Snew = np.append(Snew, Scurr[i1 + 1: i2 + 1])
        Snew = np.append(Snew, Scurr[i1])
        Snew = np.append(Snew, Scurr[i2 + 1:])
    else:
        Snew = copy.deepcopy(Scurr[0: i2])
        Snew = np.append(Snew, Scurr[i1])
        Snew = np.append(Snew, Scurr[i2: i1])
        Snew = np.append(Snew, Scurr[i1 + 1:])
    return Snew


'''
当前解经过邻域操作后得到的新解
输入Scurr：       当前解
输入pSwap：       选择交换结构的概率
输入pReversion：  选择逆转结构的概率
输入pInsertion：  选择插入结构的概率
输出Snew：        经过邻域操作后得到的的新解
'''


def Neighbor(Scurr, pSwap, pReversion, pInsertion):
    index = Roulette(pSwap, pReversion, pInsertion)
    if index == 0:
        # 交换结构
        Snew = Swap(Scurr)
    elif index == 1:
        # 逆转结构
        Snew = Reversion(Scurr)
    else:
        # 插入结构
        Snew = Insertion(Scurr)
    return Snew


'''
轮盘赌选择，输出选择邻域结构的序号
输入pSwap：       选择交换结构的概率
输入pReversion：  选择逆转结构的概率
输入pInsertion：  选择插入结构的概率
输出index：       选择所使用的邻域结构的序号，即序号：0 1 2
'''


def Roulette(pSwap, pReversion, pInsertion):
    p = np.array([pSwap, pReversion, pInsertion])
    r = np.random.rand()
    for i in range(np.size(p) - 1):
        p[i + 1] = p[i + 1] + p[i]
    for i in range(np.size(p)):
        if p[i] >= r:
            return i


if __name__ == '__main__':
    data = np.loadtxt('input.txt').astype(int)
    cap = 200
    # 提取数据信息
    vertexs = data[:, 1: 3]  # 所有点的坐标x和y
    customer = vertexs[1:, :]  # 顾客坐标
    cusnum = np.size(customer, 0)  # 顾客数
    v_num = 10  # 车辆最大允许使用数目

    demands = data[1:, 3]  # 需求量
    pdemands = data[1:, 4]  # 回收量
    h = np.array(pdist(vertexs))
    dist = np.array(squareform(h))  # 成本矩阵
    # 模拟退火参数
    belta = 100  # 违反的装载量约束的惩罚系数
    MaxOutIter = 2000  # 外层循环最大迭代次数
    MaxInIter = 300  # 里层循环最大迭代次数
    T0 = 1000  # 初始温度
    alpha = 0.99  # 冷却因子
    pSwap = 0.2  # 选择交换结构的概率
    pReversion = 0.5  # 选择逆转结构的概率
    pInsertion = 1 - pSwap - pReversion  # 选择插入结构的概率
    N = cusnum + v_num - 1  # 解长度 = 顾客数目 + 车辆最多使用数目 - 1
    # 随机构造初始解
    Scurr = np.random.permutation(N) + 1  # 随机构造初始解
    # 将初始解转换为初始配送方案
    currVC, NV, TD, violate_num, violate_cus = decode(Scurr, v_num, cusnum, cap, demands, pdemands, dist)
    # 求初始配送方案的成本=车辆行驶总距离+belta*违反的装载量约束之和
    currCost = costFuction(currVC, dist, demands, pdemands, cap, belta)
    Sbest = Scurr  # 初始将全局最优解赋值为初始解
    bestVC = currVC  # 初始将全局最优配送方案赋值为初始配送方案
    bestCost = currCost  # 初始将全局最优解的总成本赋值为初始解总成本
    BestCost = np.zeros((MaxOutIter, 1))  # 记录每一代全局最优解的总成本
    T = T0  # 温度初始化
    # 模拟退火
    for outIter in range(MaxOutIter):
        for inIter in range(MaxInIter):
            Snew = Neighbor(Scurr, pSwap, pReversion, pInsertion)  # 经过邻域结构后产生的新的解
            newVC = decode(Snew, v_num, cusnum, cap, demands, pdemands, dist)[0]  # 将新解转换为配送方案
            newCost = costFuction(newVC, dist, demands, pdemands, cap, belta)  # 求初始配送方案的成本=车辆行驶总距离+belta*违反的装载量约束之和
            # 如果新解比当前解更好，则更新当前解，以及当前解的总成本
            if newCost <= currCost:
                Scurr = Snew
                currVC = newVC
                currCost = newCost
            else:
                # 如果新解不如当前解好，则采用退火准则，以一定概率接受新解
                delta = (newCost - currCost) / currCost  # 计算新解与当前解总成本相差的百分比
                P = np.exp(-delta / T)  # 计算接受新解的概率
                # 如果0~1的随机数小于P，则接受新解，并更新当前解，以及当前解总成本
                if np.random.rand() <= P:
                    Scurr = Snew
                    currVC = newVC
                    currCost = newCost
            # 将当前解与全局最优解进行比较，如果当前解更好，则更新全局最优解，以及全局最优解总成本
            if currCost <= bestCost:
                Sbest = Scurr
                bestVC = currVC
                bestCost = currCost
        # 记录外层循环每次迭代的全局最优解的总成本
        BestCost[outIter] = bestCost
        # 显示外层循环每次迭代的信全局最优解的总成本
        print('第', str(outIter), '代全局最优解:', end="")
        bestVC, bestNV, bestTD, best_vionum, best_viocus = \
            decode(Sbest, v_num, cusnum, cap, demands, pdemands, dist)
        print('车辆使用数目：', str(bestNV), '，车辆行驶总距离：', str(bestTD),
              '，违反约束路径数目：', str(best_vionum), '，违反约束顾客数目：', str(best_viocus))
        # 更新当前温度
        T = alpha * T