import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import math
import copy
import os

'''
初始化灰狼种群
输入NIND：            种群数目
输入n：               城市数目
输入m：               旅行商数目
输入start：           起（终）点城市
输出population：      灰狼种群
'''


def init_pop(NIND, n, m, start):
    len = n + m - 1  # 个体长度
    population = np.zeros((NIND, len)).astype(int)  # 初始化种群
    for i in range(NIND):
        population[i, :] = encode(n, m, start)
    return population


'''
根据城市数目、旅行商数目以及起（终）点城市编码出灰狼个体
输入n：               城市数目
输入m：               旅行商数目
输入start：           起（终）点城市
输出individual：      灰狼个体
'''


def encode(n, m, start):
    # 生成灰狼个体的第一部分
    part1 = np.random.permutation(n)
    part1 = np.delete(part1, np.where(part1 == start))  # 将起（终）点城市从part1中删除

    # 生成灰狼个体的第二部分
    part2 = np.zeros(m).astype(int)  # 初始化每个旅行商访问城市数目（不包括start）
    if m == 1:
        part2 = n - 1
    else:
        for i in range(m):
            if i == 0:
                right = n - 1 - (m - 1)  # 最大取值
                part2[i] = np.random.randint(1, right + 1, 1, dtype=int)[0]
            elif i == m - 1:
                part2[i] = n - 1 - np.sum(part2[0: i])

            else:
                right = n - 1 - (m - i) - np.sum(part2[0: i])  # 最大取值
                part2[i] = np.random.randint(1, right + 1, 1, dtype=int)[0]
    individual = np.append(part1, part2)
    return individual


'''
计算一个种群的目标函数值
输出population：      灰狼种群
输入n：               城市数目
输入m：               旅行商数目
输入start：           起（终）点城市
输入dist：            距离矩阵
输出obj：             灰狼种群的目标函数值
'''


def obj_function(population, n, m, start, dist):
    NIND = np.size(population, 0)  # 种群数目
    obj = np.zeros((NIND, 1))  # 初始化种群目标函数值
    for i in range(NIND):
        individual = population[i, :]  # 第i个灰狼个体
        RP = decode(individual, n, m, start)  # 将第i个灰狼个体解码为旅行商行走方案
        maxETD = travel_distance(RP, dist)[2]  # 计算m个旅行商中行走距离的最大值
        obj[i] = maxETD  # 将maxETD赋值给目标函数值
    return obj


'''
对灰狼个体进行解码，解码为旅行商行走路线方案
输入individual：      灰狼个体
输入n：               城市数目
输入m：               旅行商数目
输入start：           起（终）点城市
输出RP：              旅行商行走路线方案
'''


def decode(individual, n, m, start):
    RP = np.empty(m, dtype=object)  # 初始化m条行走路线
    part1 = individual[0: n - 1]  # 提取城市排序序列
    part2 = individual[n - 1: (n + m - 1)]  # 提取各个旅行商所访问城市的数目

    for i in range(m):
        if i == 0:
            left = 0  # 在part1中，第i个旅行商访问城市的序号，即从start出发前往的下一个城市在part1中的序号
            right = part2[i]  # 在part1中，第i个旅行商访问城市的序号，即返回至start的前一个城市在part1中的序号
            route = np.insert(part1[left: right], 0, start)
            route = np.append(route, start)  # 将start添加到这条路线的首末位置
        else:
            left = np.sum(part2[0: i])  # 在part1中，第i个旅行商访问城市的序号，即从start出发前往的下一个城市在part1中的序号
            right = np.sum(part2[0: i]) + part2[i] + 1  # 在part1中，第i个旅行商访问城市的序号，即返回至start的前一个城市在part1中的序号
            route = np.insert(part1[left: right], 0, start)  # 将start添加到这条路线的首末位置
            route = np.append(route, start)
        RP[i] = route
    return RP


'''
计算所有旅行商的行走总距离、每个旅行商的行走距离、以及各旅行商的行走距离的最大值
输入RP：              旅行商行走路线方案
输入dist：            距离矩阵
输出sumTD：           所有旅行商的行走总距离
输出everyTD：         每个旅行商的行走距离
输出maxETD：          everyTD中的最大值
'''


def travel_distance(RP, dist):
    m = np.size(RP, 0)  # 旅行商数目
    everyTD = np.zeros((m, 1))  # 初始化每个旅行商的行走距离
    for i in range(m):
        route = RP[i]  # 每个旅行商的行走路线
        everyTD[i] = route_length(route, dist)

    sumTD = np.sum(everyTD)  # 所有旅行商的行走总距离
    maxETD = np.max(everyTD)  # everyTD中的最大值

    return sumTD, everyTD, maxETD


'''
计算一条路线总距离
输入route：            一条路线
输入dist：             距离矩阵
输出len：              该条路线总距离
'''


def route_length(route, dist):
    n = np.size(route)  # 这条路线所经过城市的数目，包含起点和终点城市
    len = 0
    for k in range(n - 1):
        i = route[k]
        j = route[k + 1]
        len = len + dist[i, j]
    return len


'''
对两个灰狼个体进行交叉操作
输入individual1：     灰狼个体1
输入individual2：     灰狼个体2
输入n：               城市数目
输出individual1：     交叉后的灰狼个体1
输出individual2：     交叉后的灰狼个体2
'''


def cross(individual1, individual2, n):
    individual1_ = copy.deepcopy(individual1)
    individual2_ = copy.deepcopy(individual2)
    cities_ind1 = individual1_[0: n - 1]  # 灰狼个体1的中城市序列
    cities_ind2 = individual2_[0: n - 1]  # 灰狼个体2的中城市序列
    L = n - 1  # 灰狼个体中城市序列数目

    while True:
        r1 = np.random.choice([x for x in range(L)], 1, 1)[0]
        r2 = np.random.choice([x for x in range(L)], 1, 1)[0]
        if r1 != r2:
            s = min(r1, r2)
            e = max(r1, r2)
            a0 = np.append(cities_ind2[s: e + 1], cities_ind1)
            b0 = np.append(cities_ind1[s: e + 1], cities_ind2)
            for i in range(np.size(a0)):
                aindex = np.where(a0 == a0[i])[0].ravel()[:]
                bindex = np.where(b0 == b0[i])[0].ravel()[:]
                if np.size(aindex) > 1:
                    a0 = np.delete(a0, aindex[1])
                if np.size(bindex) > 1:
                    b0 = np.delete(b0, bindex[1])

                if i == np.size(a0) - 1:
                    break

            cities_ind1 = a0
            cities_ind2 = b0
            break

    individual1_[0: n - 1] = cities_ind1
    individual2_[0: n - 1] = cities_ind2
    return individual1_, individual2_


'''
局部搜索函数
输出individual：      灰狼个体
输入n：               城市数目
输入m：               旅行商数目
输入k：               移除相邻路径的数目
输入start：           起（终）点城市
输入dist：            距离矩阵
输出individual：      局部搜索后的灰狼个体
输出ind_obj：         局部搜索后的灰狼个体的目标函数值
'''


def LocalSearch(individual, n, m, k, start, dist):
    alpha_RP = decode(individual, n, m, start)  # 将灰狼个体解码为旅行商行走方案
    alpha_TD1 = travel_distance(alpha_RP, dist)[2]  # 灰狼个体的目标函数值
    removed1, sdestroy1 = remove(alpha_RP, n, m, k, start, dist)  # 对灰狼个体进行移除操作
    s_alpha = repair(removed1, sdestroy1, dist)  # 对灰狼个体进行修复操作
    alpha_TD2 = travel_distance(s_alpha, dist)[2]  # 灰狼个体修复后的目标函数值

    # 只有目标函数值减小，才会接受新行走方案，并转换为灰狼个体
    if alpha_TD2 < alpha_TD1:
        individual = change(s_alpha)
    ind_obj = obj_function(np.array([individual]), n, m, start, dist)  # 计算individual的目标函数值
    return individual, ind_obj


'''
Adjacent String Removal根据当前解的情况，会从k条临近路径中的每条路径中移除l个城市
输入RP：              旅行商行走路线方案
输入k：               移除相邻路径的数目
输入n：               城市数目
输入m：               旅行商数目
输入start：           起（终）点城市
输入dist：            距离矩阵
输出removed：         被移出的城市集合
输出sdestroy：        移出removed中的城市后的RP
'''


def remove(RP, n, m, k, start, dist):
    avgt = int(np.floor((n - 1) / m))  # 平均每条路线上的城市数目
    removed = np.array([]).astype(int)  # 被移除城市的集合
    T = np.array([])  # 被破坏路径的集合
    iseed = int(np.ceil(np.random.rand() * (n - 1)))  # 从当前解中随机选出要被移除的城市
    lst = adj(start, iseed, dist)  # 与iseed距离由小到大的排序数组
    for i in range(np.size(lst)):
        if np.size(T) < k:
            r, rindex = tour(lst[i], RP)  # 找出城市lst(i)所在路径的序号
            fr = np.argwhere(T == rindex).ravel()  # 在破坏路径集合中查找是否有该路径
            # 如果要破坏的路径不在T中
            if np.size(fr) == 0:
                lmax = min(np.size(r) - 2, avgt)  # 从当前路线中最多移除的城市数目
                # 只有当当前路线至少经过一个城市时（不包括起点和终点），才考虑移除城市
                if lmax >= 1:
                    l = np.random.randint(0, lmax, 1, dtype=int)[0] + 1  # 计算在该条路径上移除的城市的数目
                    Rroute = String(l, lst[i], r, start)  # 从路径r中移除包含lsr(i)在内的l个连续的城市
                    removed = np.append(removed, Rroute)  # 将Rroute添加到removed中
                    T = np.append(T, rindex)  # 将破坏的路径添加到T中

    # 将removed中的城市从RP中移除
    sdestroy = dealRemove(removed, RP)
    return removed, sdestroy


'''
与iseed距离由小到大的排序数组
输入start：           起（终）点城市
输入iseed：           种子城市
输入dist：            距离矩阵
输出lst：             与iseed距离由小到大的排序数组
'''


def adj(start, iseed, dist):
    di = copy.deepcopy(dist[iseed, :])  # iseed与其他城市的距离数组
    di[start] = np.inf  # 将iseed与起点start的距离设为无穷大
    lst = np.argsort(di)  # 对di从小到大排序
    return lst


'''
找出城市i所在的路径,以及所在路径的序号
输入visit：       城市编号
输入RP：          当前行走方案
输出route：       城市visit在RP中所在的路线
输出rindex：      城市visit在RP中所在的路线序号
'''


def tour(visit, RP):
    m = np.size(RP, 0)
    for i in range(m):
        r = RP[i]
        fv = np.argwhere(r == visit).ravel()
        if np.size(fv) != 0:
            route = r
            rindex = i
            break
    return route, rindex


'''
从visit所在的路径中移除包含visit在内的连续l个城市
输入l：               要从该路径移除城市的数目
输入visit：           从该路径移除的城市
输入route：           visit所在的路径
输入start：           起（终）点城市
输出Rroute：          从当前路径中连续移除l个城市的集合
'''


def String(l, visit, route, start):
    r_copy = copy.deepcopy(route)  # 复制路径
    r_copy = np.delete(r_copy, np.argwhere(r_copy == start).ravel())  # 将start从r_copy中删除
    lr = np.size(r_copy)  # r_copy中城市数目
    findv = np.argwhere(r_copy == visit).ravel()[0]  # 找出visit在r_copy中的位置
    vLN = findv  # visit左侧的元素个数
    vRN = lr - findv  # visit右侧的元素个数
    if vLN <= vRN:
        if vLN < l - 1:
            nR = int(np.floor((l - 1 - vLN + np.random.rand() * (vRN - l + 1 + vLN))))
            nL = l - 1 - nR  # visit左侧要移除元素的数目
        if (vLN <= l - 1) and (vRN >= l - 1):
            nR = int(np.floor(l - 1 - vLN + np.random.rand() * (vLN)))
            nL = l - 1 - nR  # visit左侧要移除元素的数目
        if vLN > l - 1:
            nR = int(np.floor(np.random.rand() * vLN))  # visit右侧要移除元素的数目
            nL = l - 1 - nR  # visit左侧要移除元素的数目
        r_copy = r_copy[findv - nL:findv + nR + 1]
    if vLN > vRN:
        if vLN < l - 1:
            nL = int(np.floor(l - 1 - vRN + np.random.rand() * (vLN - l + 1 + vRN)))
            nR = l - 1 - nL  # visit右侧要移除元素的数目
        if (vRN <= l - 1) and (vLN >= l - 1):
            nL = int(np.floor(l - 1 - vRN + np.random.rand() * (vRN)))
            nR = l - 1 - nL  # visit右侧要移除元素的数目
        if vRN > l - 1:
            nL = int(np.floor(np.random.rand() * vRN))  # visit左侧要移除元素的数目
            nR = l - 1 - nL  # visit右侧要移除元素的数目
        r_copy = r_copy[findv - nL:findv + nR + 1]
    Rroute = r_copy
    return Rroute


'''
将移除集合中的元素从当前解中移除
输入removed：         被移出的城市集合
输入RP：              旅行商行走路线方案
输出sdestroy：        移出removed中的城市后的RP
'''


def dealRemove(removed, RP):
    # 将removed中的城市从VC中移除
    sdestroy = RP  # 移出removed中的城市后的RP
    nre = np.size(removed)  # 最终被移出城市的总数量
    m = np.size(RP, 0)  # 旅行商数目
    for i in range(m):
        route = RP[i]
        for j in range(nre):
            findri = np.argwhere(route == removed[j]).ravel()
            if np.size(findri) != 0:
                route = np.delete(route, findri)
        sdestroy[i] = route
    sdestroy = deal_rp(sdestroy)[0]
    return sdestroy


'''
根据RP整理出fRP，将RP中空的行走路线删除
输入RP：          行走路线方案
输出FRP：         删除空路线后的RP
输出m：           旅行商数目
'''


def deal_rp(RP):
    index = np.array([])
    for i in range(np.size(RP)):
        if np.size(RP[i]) == 0:
            index = np.append(index, i)
    if np.size(index) != 0:
        RP = np.delete(RP, index)  # 删除cell数组中的空元胞
    m = np.size(RP, 0)  # 新方案中所需旅行商的数目
    return RP, m


'''
计算将当前城市插回到当前路线中“插入成本”最小的位置
输入visit         待插入城市
输入route：       一条行走路线
输入dist：        距离矩阵
输出newRoute：    将visit插入到当前路线最佳位置后的行走路线
输出deltaC：      将visit插入到当前路线最佳位置后的插入成本
'''


def insRoute(visit, route, dist, maxETD):
    start = route[0]  # 起（终）点城市
    rcopy = copy.deepcopy(route)  # 复制路线
    rcopy = np.delete(rcopy, np.argwhere(rcopy == start).ravel())  # 将start从rcopy中删除
    lr = np.size(route) - 2  # 除去起点城市和终点城市外，当前路径上的城市数目
    # 先将城市插回到增量最小的位置
    rc0 = np.array([[-1] * (lr + 3)])  # 记录插入城市后符合约束的路径
    delta0 = np.array([[-1]])  # 记录插入城市后的增量
    for i in range(lr + 1):
        if i == lr:
            rc = np.insert(rcopy, 0, start)
            rc = np.append(rc, [visit, start])

        elif i == 0:
            rc = np.insert(rcopy, 0, [start, visit])
            rc = np.append(rc, start)
        else:
            rc = np.insert(rcopy, i - 1, visit)
            rc = np.insert(rc, 0, start)
            rc = np.append(rc, start)
        rc0 = np.vstack((rc0, rc))
        alen = route_length(rc, dist)
        dif = alen - maxETD  # 计算插入成本
        delta0 = np.vstack((delta0, dif))
    rc0 = rc0[1:]
    delta0 = delta0[1:]
    deltaC = np.min(delta0)
    ind = np.argmin(delta0)
    newRoute = rc0[ind, :]
    return newRoute, deltaC


'''
行走方案与灰狼个体之间进行转换
输入RP：          旅行商行走路线方案
输出individual：  灰狼个体
'''


def change(RP):
    m = np.size(RP, 0)  # 旅行商数目
    individual = np.array([]).astype(int)
    lr = np.zeros(m).astype(int)  # 每个旅行商所服务的城市数目
    for i in range(m):
        route = copy.deepcopy(RP[i])
        start = route[0]
        route = np.delete(route, np.argwhere(route == start).ravel())
        lr[i] = np.size(route)
        individual = np.append(individual, route)
    individual = np.append(individual, lr)
    return individual


'''
修复函数，依次将removed中的城市插回到行走方案中
先计算removed中各个城市插回当前解中所产生的“插入成本”，然后再从上述各个城市中
找出一个遗憾值(即插入成本第2小-入成本第1小)最大的城市插回，反复执行，直到全部插回
输入removed：     被移除城市的集合
输入sdestroy：    破坏后的行走路线方案
输入dist：        距离矩阵
输出srepairc：    修复后的行走方案
'''


def repair(removed, sdestroy, dist):
    srepair = sdestroy  # 初始将破坏后的解赋值给修复解
    # 反复插回removed中的城市，直到全部城市插回
    while np.size(removed):
        maxETD = travel_distance(srepair, dist)[2]  # 计算当前解的各条路线的行走距离的最大值
        nr = np.size(removed)  # 移除集合中城市数目
        ri = np.zeros((nr, 1)).astype(int)  # 存储removed各城市最好插回路径
        rid = np.zeros((nr, 1))  # 存储removed各城市插回最好插回路径后的遗憾值
        m = np.size(srepair, 0)  # 当前解的旅行商数目
        # 逐个计算将removed中的城市插回当前解中各位置后的插入成本
        for i in range(nr):
            visit = removed[i]  # 当前要插回的城市
            dec = np.array([])  # 对应于将当前城市插回到当前解各路径后的最小插入成本
            ins = np.array([])  # 记录可以插回路径的序号
            for j in range(m):
                route = srepair[j]  # 当前路径
                deltaC = insRoute(visit, route, dist, maxETD)[1]
                dec = np.append(dec, deltaC)
                ins = np.append(ins, j)
            sd = np.sort(dec)
            sdi = np.argsort(dec)  # 将dec升序排列
            insc = ins[sdi]  # 将ins的序号与dec排序后的序号对应
            ri[i] = insc[0]  # 更新当前城市最好插回路径
            if np.size(dec) > 1:
                del2 = sd[1] - sd[0]  # 计算将当前城市插回到当前解的遗憾值
                rid[i] = del2  # 更新当前城市插回最“好”插回路径后的遗憾值
            else:
                del2 = sd[0]  # 计算第2小成本增量与第1小成本增量差值
                rid[i] = del2  # 更新当前城市插回最“好”插回路径后的遗憾值

        firIns = np.argmax(rid)  # 找出遗憾值最大的城市序号
        rIns = ri[firIns]  # 插回路径序号
        # 将firIns插回到rIns
        srepair[rIns] = [np.array(insRoute(removed[firIns], srepair[rIns][0], dist, maxETD)[0])]
        removed = np.delete(removed, firIns)
    return srepair


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    # 输入数据
    dataset = np.loadtxt('input.txt')  # 数据中，每一列的含义分别为[序号，x坐标，y坐标]
    x = np.array(dataset[:, 1])  # x坐标
    y = np.array(dataset[:, 2])  # y坐标
    vertexes = np.array(dataset[:, 1:])  # 提取各个城市的xy坐标
    n = np.size(dataset, 0)  # 城市数目
    m = 2  # 旅行商数目
    start = 0  # 起点城市
    h = np.array(pdist(vertexes))  # 计算各个城市之间的距离，一共有1+2+......+(n-1)=n*(n-1)/2个
    dist = np.array(squareform(h))  # 将各个城市之间的距离转换为n行n列的距离矩阵
    # 灰狼算法参数设置
    NIND = 50  # 灰狼个体数目
    MAXGEN = 200  # 最大迭代次数
    k = m  # 移除相邻路径的数目
    # 初始化种群
    population = init_pop(NIND, n, m, start)
    init_obj = obj_function(population, n, m, start, dist)  # 初始种群目标函数值

    # 灰狼优化
    gen = 0  # 计数器
    best_alpha = np.zeros((MAXGEN, n + m - 1)).astype(int)  # 记录每次迭代过程中全局最优灰狼个体
    best_obj = np.zeros((MAXGEN, 1))  # 记录每次迭代过程中全局最优灰狼个体的目标函数值
    alpha_individual = population[0, :]  # 初始灰狼α个体
    alpha_obj = init_obj[0]  # 初始灰狼α的目标函数值
    beta_individual = population[1, :]  # 初始灰狼β个体
    beta_obj = init_obj[1]  # 初始灰狼β的目标函数值
    delta_individual = population[2, :]  # 初始灰狼δ个体
    delta_obj = init_obj[2]  # 初始灰狼δ的目标函数值

    while gen < MAXGEN:

        obj = obj_function(population, n, m, start, dist)  # 计算灰狼种群目标函数值

        # 确定当前种群中的灰狼α个体、灰狼β个体和灰狼δ个体
        for i in range(NIND):
            # 更新灰狼α个体
            if obj[i, 0] < alpha_obj:
                alpha_obj = obj[i, 0]
                alpha_individual = population[i, :]
            # 更新灰狼β个体
            if (obj[i, 0] > alpha_obj) and (obj[i, 0] < beta_obj):
                beta_obj = obj[i, 0]
                beta_individual = population[i, :]
            # 更新灰狼δ个体
            if (obj[i, 0] > alpha_obj) and (obj[i, 0] > beta_obj) and (obj[i, 0] < delta_obj):
                delta_obj = obj[i, 0]
                delta_individual = population[i, :]
        for i in range(NIND):
            r = np.random.rand()
            individual = population[i, :]  # 第i个灰狼个体

            # 概率更新灰狼个体位置
            # 概率更新灰狼个体位置
            if r <= (1 / 3):
                new_individual = cross(individual, alpha_individual, n)[0]
            elif r <= (2 / 3):
                new_individual = cross(individual, beta_individual, n)[0]
            else:
                new_individual = cross(individual, delta_individual, n)[0]

            population[i, :] = new_individual
        # 局部搜索操作

        alpha_individual, alpha_obj = LocalSearch(alpha_individual, n, m, k, start, dist)
        beta_individual, beta_obj = LocalSearch(beta_individual, n, m, k, start, dist)
        delta_individual, delta_obj = LocalSearch(delta_individual, n, m, k, start, dist)

        # 记录全局最优灰狼个体
        best_alpha[gen, :] = alpha_individual  # 记录全局最优灰狼个体
        best_obj[gen, 0] = alpha_obj  # 记录全局最优灰狼个体的目标函数值
        # 打印当前代数全局最优解
        print('第', str(gen), '代最优解的目标函数值：', str(alpha_obj))
        # 更新计数器
        gen = gen + 1  # 计数器加1
    # 打印每次迭代的全局最优灰狼个体的目标函数值变化趋势图
    '''
    打印函数未写
    '''

    # 将全局最优灰狼个体解码为旅行商行走路线方案
    bestRP = decode(alpha_individual, n, m, start)  # 全局最优灰狼个体解码为旅行商行走方案
    bestTD, bestETD, bestMETD = travel_distance(bestRP, dist)  # 全局最优灰狼个体的目标函数值
    print(bestTD, bestETD, bestMETD)