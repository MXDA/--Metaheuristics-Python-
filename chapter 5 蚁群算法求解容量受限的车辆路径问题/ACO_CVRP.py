import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import math
import copy
import os

'''
找到蚂蚁k从i点出发可以移动到的下一个点j的集合，j点必须是满足容量且是未被蚂蚁k服务过的顾客
输入k：                   蚂蚁序号
输入Table：               路径记录表
输入cap：                 最大装载量
输入demands：             顾客需求量
输入dist：                距离矩阵
输出Nik：                 蚂蚁k从i点出发可以移动到的下一个点j的集合，j点必须是满足容量及时间约束且是未被蚂蚁k服务过的顾客
'''


def next_point_set(k, Table, cap, demands, dist):
    route_k = copy.deepcopy(Table[k, :])  # 蚂蚁k的路径
    cusnum = np.size(Table, 1)  # 顾客数目
    route_k = np.delete(route_k, np.argwhere(route_k == 0).ravel())  # 将0从蚂蚁k的路径记录数组中删除
    # 如果蚂蚁k已经访问了若干个顾客
    if np.size(route_k) != 0:
        VC = decode(route_k, cap, demands, dist)[0]  # 蚂蚁k目前为止所构建出的所有路径
        route = VC[-1]  # 蚂蚁k当前正在构建的路径
        lr = np.size(route)  # 蚂蚁k当前正在构建的路径所访问顾客数目
        preroute = np.zeros(lr + 1).astype(int)  # 临时变量，储存蚂蚁k当前正在构建的路径添加下一个点后的路径
        preroute[0: lr] = route  # 临时变量，储存蚂蚁k当前正在构建的路径添加下一个点后的路径
        allSet = np.array([x + 1 for x in range(cusnum)])  # setxor(a,b)可以得到a,b两个矩阵不相同的元素，也叫不在交集中的元素，
        unVisit = np.setxor1d(route_k, allSet)  # 找出蚂蚁k未服务的顾客集合
        uvNum = np.size(unVisit)  # 找出蚂蚁k未服务的顾客数目
        Nik = np.zeros(uvNum).astype(int)  # 初始化蚂蚁k从i点出发可以移动到的下一个点j的集合，j点必须是满足容量及时间约束且是未被蚂蚁k服务过的顾客
        for i in range(uvNum):
            preroute[-1] = unVisit[i]  # 将unVisit(i)添加到蚂蚁k当前正在构建的路径route后
            flag = JudgeRoute(preroute, demands, cap)  # 判断一条路线是否满足装载量约束，1表示满足，0表示不满足
            # 如果满足约束，则将unVisit(i)添加到蚂蚁k从i点出发可以移动到的下一个点j的集合中
            if flag == 1:
                Nik[i] = unVisit[i]
        Nik = np.delete(Nik, np.argwhere(Nik == 0).ravel())  # 将0从np_set中删除
    else:
        # 如果蚂蚁k没有访问任何顾客
        Nik = np.array([x + 1 for x in range(cusnum)])  # 则所有顾客都可以成为候选点
    return Nik


'''
根据转移公式，找到蚂蚁k从i点出发移动到的下一个点j，j点必须是满足容量及时间约束且是未被蚂蚁k服务过的顾客
输入k：                   蚂蚁序号
输入Table：               路径记录表
输入Tau：                 信息素矩阵
输入Eta：                 启发函数，即距离矩阵的倒数
输入alpha：               信息素重要程度因子
输入beta：                启发函数重要程度因子
输入dist：                距离矩阵
输入cap：                 最大装载量
输入demands：             需求量
输出j：                   蚂蚁k从i点出发移动到的下一个点j
'''


def next_point(k, Table, Tau, Eta, alpha, beta, dist, cap, demands):
    route_k = copy.deepcopy(Table[k, :])  # 蚂蚁k的路径
    # 蚂蚁k正在访问的顾客编号
    i = 0
    for k in route_k:
        if (k != 0) and (k != 1):
            i = k
    route_k = np.delete(route_k, np.argwhere(route_k == 0).ravel())  # 将0从蚂蚁k的路径记录数组中删除
    cusnum = np.size(Table, 1)  # 顾客数目
    allSet = np.array([x + 1 for x in range(cusnum)])  # setxor(a,b)可以得到a,b两个矩阵不相同的元素，也叫不在交集中的元素，
    unVisit = np.setxor1d(route_k, allSet)  # 找出蚂蚁k未服务的顾客集合
    uvNum = np.size(unVisit)  # 找出蚂蚁k未服务的顾客数目
    VC = decode(route_k, cap, demands, dist)[0]  # 蚂蚁k目前为止所构建出的所有路径
    # 如果当前路径配送方案为空
    if np.size(VC) != 0:
        route = VC[-1]  # 蚂蚁k当前正在构建的路径
    else:
        # 如果当前路径配送方案为空
        route = np.array([])
    lr = np.size(route)  # 蚂蚁k当前正在构建的路径所访问顾客数目
    preroute = np.zeros(lr + 1).astype(int)  # 临时变量，储存蚂蚁k当前正在构建的路径添加下一个点后的路径
    preroute[0: lr] = route
    Nik = next_point_set(k, Table, cap, demands, dist)  # 找到蚂蚁k从i点出发可以移动到的下一个点j的集合，j点必须是满足容量且是未被蚂蚁k服务过的顾客
    # 如果r>r0，依据概率公式用轮盘赌法选择点j
    # 如果Nik非空，即蚂蚁k可以在当前路径从顾客i继续访问顾客
    if np.size(Nik):
        Nik_num = np.size(Nik)
        p_value = np.zeros(Nik_num)  # 记录状态转移概率
        for h in range(Nik_num):
            j = Nik[h]
            p_value[h] = ((Tau[i, j]) ** alpha) * ((Eta[i, j]) ** beta)
        p_value = p_value / np.sum(p_value)
        index = roulette(p_value)  # 根据轮盘赌选出序号
        j = Nik[index]  # 确定顾客j
    else:
        # 如果Nik为空，即蚂蚁k必须返回配送中心，从配送中心开始访问新的顾客
        p_value = np.zeros(uvNum)  # 记录状态转移概率
        for h in range(uvNum):
            j = unVisit[h]
            p_value[h] = ((Tau[i, j]) ** alpha) * ((Eta[i, j]) ** beta)
        p_value = p_value / np.sum(p_value)
        index = roulette(p_value)  # 根据轮盘赌选出序号
        j = unVisit[index]  # 确定顾客j
    return j


'''
将蚂蚁构建的完整路径转换为配送方案
输入route_k：             蚂蚁k构建的完整路径
输入cap：                 最大装载量
输入demands：             需求量
输入dist：                距离矩阵，满足三角关系，暂用距离表示花费c[i][j]=dist[i][j]
输出VC：                  配送方案，每辆车所经过的顾客
输出NV：                  车辆使用数目
输出TD：                  车辆行驶总距离

思路：例子：当前一只蚂蚁构建的完整路径为53214，
那么首先从头开始遍历，第一条路径为5，然后依次将3添加到这条路径，
则该条路径变为53，此时要检验53这条路径是否满足时间窗约束和装载量约束，
如不满足其中任何一个约束，则需要新建路径，则3为一个顾客，然后按照这种方法添加。
如果满足上述两个约束，则继续将2添加到53这条路径，然后继续检验532这条路径是否满足时间窗约束和装载量约束，
依此类推。
'''


def decode(route, cap, demands, dist):
    route_k = copy.deepcopy(route)
    route_k = np.delete(route_k, np.argwhere(route_k == 0).ravel())  # 将0从蚂蚁k的路径记录数组中删除
    cusnum = np.size(route_k)  # 已服务的顾客数目
    VC = np.empty(cusnum, dtype=object)  # 每辆车所经过的顾客
    count = 0  # 车辆计数器，表示当前车辆使用数目
    preroute = np.array([]).astype(int)  # 存放某一条路径
    for i in range(cusnum):
        preroute = np.append(preroute, route_k[i])  # 将第route_k(i)添加到路径中
        flag = JudgeRoute(preroute, demands, cap)  # 判断一条路线是否满足装载量约束，1表示满足，0表示不满足
        if flag == 1:
            # 如果满足约束，则更新车辆配送方案VC
            VC[count] = preroute
        else:
            # 如果满足约束，则清空preroute，并使count加1
            preroute = np.array([route_k[i]])
            count = count + 1
            VC[count] = preroute
    VC = deal_vehicles_customer(VC)[0]  # 将VC中空的数组移除
    NV = np.size((VC, 0))  # 车辆使用数目
    TD = travel_distance(VC, dist)[0]
    return VC, NV, TD


'''
根据VC整理出FVC，将VC中空的配送路线删除
输入VC：          配送方案，即每辆车所经过的顾客
输出FVC：         删除空配送路线后的VC
输出NV：          车辆使用数目
'''


def deal_vehicles_customer(VC):
    index = np.array([]).astype(int)
    for i in range(np.size(VC)):
        if VC[i] is None:
            index = np.append(index, i)
    if np.size(index) != 0:
        VC = np.delete(VC, index)  # 删除cell数组中的空元胞
    m = np.size(VC, 0)  # 新方案中所需旅行商的数目
    return VC, m


'''
计算一条路线总距离
输入route：            一条配送路线
输入dist：             距离矩阵
输出p_l：              该条路线总距离
'''


def part_length(route, dist):
    n = np.size(route)
    p_l = 0.0
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
判断一条路线是否满足装载量约束，1表示满足，0表示不满足
输入route：       一条配送路线
输入demands：     顾客需求量
输入cap：         车辆最大装载量
输出flagR：       标记一条路线是否满足装载量约束，1表示满足，0表示不满足
'''


def JudgeRoute(route, demands, cap):
    flagR = 1  # 初始满足装载量约束
    Ld = leave_load(route, demands)  # 计算该条路径上离开配送中心时的载货量
    # 如果不满足装载量约束，则将flagR赋值为0
    if Ld > cap:
        flagR = 0
    return flagR


'''
计算某一条路径上离开集配中心和顾客时的装载量
输入route：               一条配送路线
输入demands：             顾客需求量
输出Ld：                  货车离开配送中心时的装载量
'''


def leave_load(route, demands):
    n = np.size(route)  # 配送路线经过顾客的总数目
    Ld = 0  # 初始车辆在配送中心时的装货量为0
    if n != 0:
        for i in range(n):
            if route[i] != 0:
                Ld = Ld + demands[route[i] - 1]
    return Ld


'''
计算一个配送方案的总成本=车辆行驶总距离
输入VC：          配送方案
输入dist：        距离矩阵
输出cost：        该配送方案的总成本
输出NV：          车辆使用数目
'''


def costFun(VC, dist):
    NV = np.size(VC)  # 车辆使用数目
    TD = travel_distance(VC, dist)[0]  # 行驶总距离
    cost = TD
    return cost, NV, TD


'''
轮盘赌
输入p_value：                 下一个访问点集合中每一个点的状态转移概率
输出index：                   轮盘赌选择的p_value的行序号
'''


def roulette(p_value):
    r = np.random.rand()
    c = copy.deepcopy(p_value)
    for i in range(np.size(c) - 1):
        c[i + 1] += c[i]
    for i in range(np.size(c)):
        if c[i] >= r:
            return i


'''
更新路径R的信息素
输入Tau：                 更新前的信息素矩阵
输入bestR：               最优蚂蚁所构建的完整路径
输入rho：                 信息素挥发因子
输入Q：                   蚂蚁构建一次完整路径所释放的信息素总量
输入cap：                 最大装载量
输入demands：             需求量
输入dist：                距离矩阵
输出Tau1：                 更新后的信息素矩阵
'''


def updateTau(Tau, bestR, rho, Q, cap, demands, dist):
    bestTD = decode(bestR, cap, demands, dist)[2]
    cusnum = np.size(dist, 0) - 1
    Delta_Tau = np.zeros((cusnum + 1, cusnum + 1))
    delta_Tau = Q / bestTD
    Tau1 = copy.deepcopy(Tau)
    for j in range(cusnum - 1):
        Delta_Tau[bestR[j], bestR[j + 1]] = Delta_Tau[bestR[j], bestR[j + 1]] + delta_Tau
        Tau1[bestR[j], bestR[j + 1]] = rho * Tau1[bestR[j], bestR[j + 1]] + Delta_Tau[bestR[j], bestR[j + 1]]

    Delta_Tau[bestR[cusnum - 1], 0] = Delta_Tau[bestR[cusnum - 1], 0] + delta_Tau
    Tau1[bestR[cusnum - 1], 0] = rho * Tau1[bestR[cusnum - 1], 0] + Delta_Tau[bestR[cusnum - 1], 0]
    return Tau1

if __name__ == '__main__':
    dataset = np.loadtxt('input.txt').astype(int)
    cap = 200
    # 提取数据信息
    vertexs = dataset[:, 1: 3]  # 所有点的坐标x和y
    customer = vertexs[1:, :]  # 顾客坐标
    cusnum = np.size(customer, 0)  # 顾客数
    demands = dataset[1:, 3]  # 需求量
    h = np.array(pdist(vertexs))
    dist = np.array(squareform(h))  # 成本矩阵
    # 初始化参数
    m = 50  # 蚂蚁数量
    alpha = 2  # 信息素重要程度因子
    beta = 6  # 启发函数重要程度因子
    rho = 0.95  # 信息素挥发因子
    Q = 7  # 更新信息素浓度的常数
    Eta = 1 / dist  # 启发函数
    Tau = np.ones((cusnum + 1, cusnum + 1))  # 信息素矩阵
    Table = np.zeros((m, cusnum)).astype(int)  # 路径记录表
    iter = 0  # 迭代次数初值
    iter_max = 100  # 最大迭代次数
    Route_best = np.zeros((iter_max, cusnum)).astype(int)  # 各代最佳路径
    Cost_best = np.zeros((iter_max, 1))  # 各代最佳路径的成本
    # 迭代寻找最佳路径
    while iter < iter_max:
        # 先构建出所有蚂蚁的路径
        for i in range(m):
            # 逐个顾客选择
            for j in range(cusnum):
                NP = next_point(i, Table, Tau, Eta, alpha, beta, dist, cap, demands)
                Table[i, j] = NP

        # 计算各个蚂蚁的成本=1000*车辆使用数目+车辆行驶总距离
        cost = np.zeros((m, 1))
        NV = np.zeros((m, 1))
        TD = np.zeros((m, 1))
        for i in range(m):
            VC = decode(Table[i, :], cap, demands, dist)[0]
            cost[i, 0], NV[i, 0], TD[i, 0] = costFun(VC, dist)
        # 计算最小成本及平均成本
        if iter == 0:
            min_Cost = np.min(cost)
            min_index = np.argmin(cost)
            Cost_best[iter] = min_Cost
            Route_best[iter, :] = Table[min_index, :]
        else:
            min_Cost = np.min(cost)
            min_index = np.argmin(cost)
            Cost_best[iter] = min(Cost_best[iter - 1], min_Cost)
            if Cost_best[iter] == min_Cost:
                Route_best[iter, :] = Table[min_index, :]
            else:
                Route_best[iter, :] = Route_best[(iter - 1), :]
        # 更新信息素
        bestR = Route_best[iter, :]
        bestVC, bestNV, bestTD = decode(bestR, cap, demands, dist)
        Tau = updateTau(Tau, bestR, rho, Q, cap, demands, dist)
        # 打印当前最优解
        print('第', str(iter), '代最优解:')
        print('车辆使用数目：', str(bestNV), '，车辆行驶总距离：', str(bestTD))
        iter = iter + 1
        Table = np.zeros((m, cusnum)).astype(int)
    bestRoute = Route_best[-1, :]
    bestVC, bestNV, bestTD = decode(bestRoute, cap, demands, dist)
    # 画图
