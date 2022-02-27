import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import math
import copy
import os

'''
贪婪算法构造TSP的初始解
输入dist：             距离矩阵 
输出init_route：      贪婪算法构造的初始路线
输出init_len：        init_route的总距离  
'''


def construct_route(dist):
    N = np.size(dist, 0)  # 城市数目
    # 先将距离矩阵主对角线上的0赋值为无穷大
    for i in range(N):
        for j in range(N):
            if i == j:
                dist[i, j] = np.inf
    unvisited = np.array([x for x in range(N)])  # 初始未被安排的城市集合
    visited = np.array([]).astype(int)  # 初始已被安排的城市集合
    min_dist = np.min(dist)  # 找出距离矩阵中的最小值
    row, col = np.array(np.where(dist == min_dist))[0].ravel()[:]  # 在dist中找出min_dist所对应的行和列
    first = row  # 将min_dist在dist中所对应行序号作为起点
    unvisited = np.delete(unvisited, np.argwhere(unvisited == first).ravel()[0])  # 将first从unvisit中删除
    visited = np.append(visited, first)  # 把first添加到visit中
    pre_point = first  # 将first赋值给pre_point

    while np.size(unvisited) != 0:
        pre_dist = copy.deepcopy(dist[pre_point, :])  # pre_point与其它城市的距离
        pre_dist[visited] = np.inf  # 将pre_point与已经添加进来的城市之间的距离设位无穷大
        pre_point = np.argmin(pre_dist)  # 找出pre_dist中的最小值
        unvisited = np.delete(unvisited, np.argwhere(unvisited == pre_point))  # 将pre_point从unvisit中删除
        visited = np.append(visited, pre_point)  # 把pre_point添加到visit中
    init_route = visited

    init_len = route_length(init_route, dist)  # 计算init_route的总距离
    return init_route, init_len


'''
计算一条路线总距离
输入route：            一条路线
输入dist：             距离矩阵
输出len：              该条路线总距离
'''


def route_length(route, dist):
    N = np.size(route)
    route = np.append(route, route[0])
    len = 0
    for k in range(N):
        i = route[k]
        j = route[k + 1]
        len = len + dist[i, j]
    return len


'''
交换操作
比如说有6个城市，当前解为123456，随机选择两个位置，然后将这两个位置上的元素进行交换。
比如说，交换2和5两个位置上的元素，则交换后的解为153426。
输入route1：          路线1
输入i,j：             两个交换点
输出route2：          经过交换操作变换后的路线2
'''


def swap(route1, i, j):
    route2 = copy.deepcopy(route1)
    route2[i], route2[j] = route1[j], route1[i]
    return route2


'''
逆转操作
有6个城市，当前解为123456，我们随机选择两个位置，然后将这两个位置之间的元素进行逆序排列。
比如说，逆转2和5之间的所有元素，则逆转后的解为154326。
输入route1：          路线1
输入i,j：             逆转点i,j
输出route2：          经过逆转结构变换后的路线2
'''


def reversion(route1, i, j):
    i1 = min(i, j)
    i2 = max(i, j)
    route2 = copy.deepcopy(route1)
    route2[i1: i2 + 1] = np.flipud(route1[i1: i2 + 1])
    return route2


'''
插入操作
有6个城市，当前解为123456，我们随机选择两个位置，然后将这第一个位置上的元素插入到第二个元素后面。
比如说，第一个选择5这个位置，第二个选择2这个位置，则插入后的解为125346。
输入route1：          路线1
输入i,j：             插入点i,j
输出route2：          经过插入结构变换后的路线2
'''


def insertion(route1, i, j):
    if i < j:
        route2 = copy.deepcopy(route1[0: i])
        route2 = np.append(route2, route1[i + 1: j + 1])
        route2 = np.append(route2, route1[i])
        route2 = np.append(route2, route1[j + 1:])
    else:
        route2 = copy.deepcopy(route1[0: j + 1])
        route2 = np.append(route2, route1[i])
        route2 = np.append(route2, route1[j + 1: i])
        route2 = np.append(route2, route1[i + 1:])
    return route2


'''
计算swap操作后与操作前路线route的总距离的差值
输入route：            一条路线
输入dist：             距离矩阵
输入i，j：             交换点i,j
输出delta1：           swap后路线的总距离-swap前路线的总距离
'''


def cal_delta1(route, dist, i, j):
    N = np.size(route)  # 城市数目
    if (i == 0) and (j == N - 1):
        delta1 = -(dist[route[i], route[i + 1]] + dist[route[j - 1], route[j]]) + \
                 (dist[route[j], route[i + 1]] + dist[route[j - 1], route[i]])
    elif (i == 0) and (j == 1):
        delta1 = -(dist[route[N - 1], route[i]] + dist[route[j], route[j + 1]]) + \
                 (dist[route[N - 1], route[j]] + dist[route[i], route[j + 1]])
    elif i == 0:
        delta1 = -(dist[route[N - 1], route[i]] + dist[route[i], route[i + 1]] +
                   dist[route[j - 1], route[j]] + dist[route[j], route[j + 1]]) + \
                 (dist[route[N - 1], route[j]] + dist[route[j], route[i + 1]] +
                  dist[route[j - 1], route[i]] + dist[route[i], route[j + 1]])
    elif (i == N - 2) and (j == N - 1):
        delta1 = -(dist[route[i - 1], route[i]] + dist[route[j], route[0]]) + \
                 (dist[route[i - 1], route[j]] + dist[route[i], route[0]])
    elif j == N - 1:
        delta1 = -(dist[route[i - 1], route[i]] + dist[route[i], route[i + 1]] +
                   dist[route[j - 1], route[j]] + dist[route[j], route[0]]) + \
                 (dist[route[i - 1], route[j]] + dist[route[j], route[i + 1]] +
                  dist[route[j - 1], route[i]] + dist[route[i], route[0]])
    elif np.abs(i - j) == 1:
        delta1 = -(dist[route[i - 1], route[i]] + dist[route[j], route[j + 1]]) + \
                 (dist[route[i - 1], route[j]] + dist[route[i], route[j + 1]])
    else:
        delta1 = -(dist[route[i - 1], route[i]] + dist[route[i], route[i + 1]] +
                   dist[route[j - 1], route[j]] + dist[route[j], route[j + 1]]) + \
                 (dist[route[i - 1], route[j]] + dist[route[j], route[i + 1]] +
                  dist[route[j - 1], route[i]] + dist[route[i], route[j + 1]])
    return delta1


'''
将给定的route序列在i和j位置之间进行逆序排列，然后计算转换序列前和转换序列后的路径距离的差值
输入route：            一条路线
输入dist：             距离矩阵
输入i，j：             逆转点i,j
输出delta2：           reversion后路线的总距离-reversion前路线的总距离
'''


def cal_delta2(route, dist, i, j):
    N = np.size(route)  # 城市个数
    if i == 0:
        if j == N - 1:
            delta2 = 0
        else:
            delta2 = -dist[route[j], route[j + 1]] - dist[route[N - 1], route[i]] + \
                     dist[route[i], route[j + 1]] + dist[route[N - 1], route[j]]
    else:
        if j == N - 1:
            delta2 = -dist[route[i - 1], route[i]] - dist[route[0], route[j]] + \
                     dist[route[i - 1], route[j]] + dist[route[i], route[0]]
        else:
            delta2 = -dist[route[i - 1], route[i]] - dist[route[j], route[j + 1]] + \
                     dist[route[i - 1], route[j]] + dist[route[i], route[j + 1]]
    return delta2


'''
计算insertion操作后与操作前路线route的总距离的差值
输入route：            一条路线
输入dist：             距离矩阵
输入i，j：             逆转点i,j
输出delta1：           insertion后路线的总距离-insertion前路线的总距离
'''


def cal_delta3(route, dist, i, j):
    N = np.size(route)  # 城市数目
    if i < j:
        if (i == 0) and (j == N - 1):
            delta3 = 0
        elif (i == 0) and (j == 1):
            delta3 = -(dist[route[N - 1], route[i]] + dist[route[j], route[j + 1]]) + \
                     (dist[route[N - 1], route[j]] + dist[route[i], route[j + 1]])
        elif i == 0:
            delta3 = -(dist[route[N - 1], route[i]] + dist[route[i], route[i + 1]] +
                       dist[route[j], route[j + 1]]) + \
                     (dist[route[N - 1], route[i + 1]] + dist[route[j], route[i]] +
                      dist[route[i], route[j + 1]])
        elif (i == N - 2) and (j == N - 1):
            delta3 = -(dist[route[i - 1], route[i]] + dist[route[j], route[0]]) + \
                     (dist[route[i - 1], route[j]] + dist[route[i], route[0]])
        elif j == N - 1:
            delta3 = -(dist[route[i - 1], route[i]] + dist[route[i], route[i + 1]] +
                       dist[route[j], route[0]]) + (dist[route[i - 1], route[i + 1]] +
                                                    dist[route[j], route[i]] + dist[route[i], route[0]])
        elif (j - i) == 1:
            delta3 = -(dist[route[i - 1], route[i]] + dist[route[j], route[j + 1]]) + \
                     (dist[route[i - 1], route[j]] + dist[route[i], route[j + 1]])
        else:
            delta3 = -(dist[route[i - 1], route[i]] + dist[route[i], route[i + 1]] +
                       dist[route[j], route[j + 1]]) + \
                     (dist[route[i - 1], route[i + 1]] + dist[route[j], route[i]] +
                      dist[route[i], route[j + 1]])
    else:
        if (i == N - 1) and (j == 0):
            delta3 = -(dist[route[i - 1], route[i]] + dist[route[j], route[j + 1]]) + \
                     (dist[route[i - 1], route[j]] + dist[route[i], route[j + 1]])
        elif (i - j) == 1:
            delta3 = 0
        elif i == N - 1:
            delta3 = -(dist[route[i - 1], route[i]] + dist[route[i], route[0]]
                       + dist[route[j], route[j + 1]]) + (dist[route[i - 1], route[0]]
                                                          + dist[route[j], route[i]] +
                                                          dist[route[i], route[j + 1]])

        else:
            delta3 = -(dist[route[i - 1], route[i]] + dist[route[i], route[i + 1]]
                       + dist[route[j], route[j + 1]]) + (dist[route[i - 1], route[i + 1]] +
                                                          dist[route[j], route[i]] +
                                                          dist[route[i], route[j + 1]])

    return delta3


'''
swap操作后生成新的距离差矩阵Delta
输入route：            一条路线
输入dist：             距离矩阵
输入i，j：             交换点i,j
输出Delta1：           swap操作后的距离差值的矩阵
'''


def Update1(route, dist, i, j):
    N = np.size(route)  # 城市个数
    route2 = swap(route, i, j)  # 交换route上i和j两个位置上的城市
    Delta1 = np.zeros((N, N))  # N行N列的Delta初始化，每个位置上的元素是距离差值
    for i in range(N - 1):
        for j in range(i + 1, N):
            Delta1[i, j] = cal_delta1(route2, dist, i, j)
    return Delta1


'''
reversion操作后生成新的距离差矩阵Delta
输入route：            一条路线
输入dist：             距离矩阵
输入i，j：             逆转点i,j
输出Delta2：           reversion操作后的距离差值的矩阵
'''


def Update2(route, dist, i, j):
    N = np.size(route)  # 城市个数
    route2 = reversion(route, i, j)  # 逆转route上i和j两个位置上的城市
    Delta2 = np.zeros((N, N))  # N行N列的Delta初始化，每个位置上的元素是距离差值
    for i in range(N - 1):
        for j in range(j + 1, N):
            Delta2[i, j] = cal_delta2(route2, dist, i, j)
    return Delta2


'''
insertion操作后生成新的距离差矩阵Delta
输入route：            一条路线
输入dist：             距离矩阵
输入i，j：             插入点i,j
输出Delta1：           insertion操作后的距离差值的矩阵
'''


def Update3(route, dist, i, j):
    N = np.size(route)  # 城市个数
    route2 = insertion(route, i, j)  # 插入route上i和j两个位置上的城市
    Delta3 = np.zeros((N, N))  # N行N列的Delta初始化，每个位置上的元素是距离差值
    for i in range(N):
        for j in range(N):
            if i != j:
                Delta3[i, j] = cal_delta3(route2, dist, i, j)

    return Delta3


'''
扰动，随机选择当前邻域中的一个解更新当前解
输入route：           一条路线
输入dist：            距离矩阵
输入k：               当前邻域序号
输出route_shake：     扰动操作后得到的路线
输出len_shake：       该条路线的距离
'''


def shaking(route, dist, k):
    N = np.size(route)  # 城市数目
    i, j = random.sample(range(N), 2)  # 随机选择进行操作的两个点的序号
    if k == 1:
        route_shake = swap(route, i, j)
    elif k == 2:
        route_shake = reversion(route, i, j)
    else:
        route_shake = insertion(route, i, j)
    len_shake = route_length(route_shake, dist)
    return route_shake, len_shake


'''
对route不断进行交换操作后所得到的路线以及所对应的总距离
输入route：           一条路线
输入dist：            距离矩阵
输入M：               最多进行邻域操作的次数
输出swap_route：      对route不断进行交换操作后所得到的路线
输出swap_len：        swap_route的总距离
'''


def swap_neighbor(route, dist, M):
    N = np.size(route)  # 城市数目
    # print("***")
    Delta1 = np.zeros((N, N))  # 交换任意两个位置之间序列的元素所产的距离差的矩阵
    for i in range(N - 1):
        for j in range(i + 1, N):
            Delta1[i, j] = cal_delta1(route, dist, i, j)
    cur_route = copy.deepcopy(route)  # 初始化当前路线
    m = 1  # 初始化计数器
    while m <= M:
        min_value = np.min(Delta1)  # 找出距离差值矩阵中最小的距离差值
        # 如果min_value小于0，才能更新当前路线和距离矩阵。否则，终止循环
        if min_value < 0:
            min_row, min_col = np.argwhere(Delta1 == min_value)[0].ravel()  # 找出距离差值矩阵中最小的距离差值所对应的行和列
            Delta1 = Update1(cur_route, dist, min_row, min_col)  # 更新距离差值矩阵
            # print(route_length(cur_route, dist) + min_value, "--358--")
            cur_route = swap(cur_route, min_row, min_col)  # 更新当前路线
        # print(route_length(cur_route, dist), "--360--")
        else:
            break
        m = m + 1
    swap_route = cur_route  # 将当前路线cur_route赋值给swap_route
    swap_len = route_length(swap_route, dist)  # swap_route的总距离
    # print(swap_len, "--366--")
    # print("***")
    return swap_route, swap_len


'''
对route不断进行逆转操作后所得到的路线以及所对应的总距离
输入route：           一条路线
输入dist：            距离矩阵
输入M：               最多进行M次邻域操作
输出reversion_route： 对route不断进行逆转操作后所得到的路线
输出reversion_len：   reversion_route的总距离
'''


def reversion_neighbor(route, dist, M):
   # print("***")
    N = np.size(route)  # 城市数目
    Delta2 = np.zeros((N, N))  # 逆转任意两个位置之间序列的元素所产的距离差的矩阵
    for i in range(N - 1):
        for j in range(i + 1, N):
            Delta2[i, j] = cal_delta2(route, dist, i, j)
    cur_route = copy.deepcopy(route)  # 初始化当前路线
    m = 1  # 初始化计数器
    while m <= M:
        min_value = np.min(Delta2)  # 找出距离差值矩阵中最小的距离差值
        # 如果min_value小于0，才能更新当前路线和距离矩阵。否则，终止循环
        if min_value < 0:
            min_row, min_col = np.argwhere(Delta2 == min_value)[0].ravel()  # 找出距离差值矩阵中最小的距离差值所对应的行和列
            Delta2 = Update2(cur_route, dist, min_row, min_col)  # 更新距离差值矩阵
           # print(route_length(cur_route, dist) + min_value, "--396--")
            cur_route = reversion(cur_route, min_row, min_col)  # 更新当前路线
            #print(route_length(cur_route, dist), "--398--")
        else:
            break
        m = m + 1
    reversion_route = cur_route  # 将当前路线cur_route赋值给reversion_route
    reversion_len = route_length(reversion_route, dist)  # reversion_route的总距离
    #print("***")
    return reversion_route, reversion_len


'''
对route不断进行插入操作后所得到的路线以及所对应的总距离
输入route：           一条路线
输入dist：            距离矩阵
输入M：               最多进行M次邻域操作
输出insertion_route： 对route不断进行插入操作后所得到的路线
输出insertion_len：   insertion_route的总距离
'''


def insertion_neighbor(route, dist, M):
    N = np.size(route)  # 城市数目
    Delta3 = np.zeros((N, N))  # 逆转任意两个位置之间序列的元素所产的距离差的矩阵
    for i in range(N - 1):
        for j in range(i + 1, N):
            Delta3[i, j] = cal_delta3(route, dist, i, j)
    cur_route = copy.deepcopy(route)  # 初始化当前路线
    m = 1  # 初始化计数器
    while m <= M:
        min_value = np.min(Delta3)  # 找出距离差值矩阵中最小的距离差值
        # 如果min_value小于0，才能更新当前路线和距离矩阵。否则，终止循环
        if min_value < 0:
            min_row, min_col = np.argwhere(Delta3 == min_value)[0].ravel()  # 找出距离差值矩阵中最小的距离差值所对应的行和列
            Delta3 = Update3(cur_route, dist, min_row, min_col)  # 更新距离差值矩阵
            cur_route = insertion(cur_route, min_row, min_col)  # 更新当前路线
        else:
            break
        m = m + 1
    insertion_route = cur_route  # 将当前路线cur_route赋值给reversion_route
    insertion_len = route_length(insertion_route, dist)  # reversion_route的总距离
    return insertion_route, insertion_len


'''
TSP路线可视化
输入route：           一条路线
输入x,y：             x,y坐标
'''


def plot_route(route, x, y):
    plt.figure(2)
    route = np.append(route, route[0])
    plt.plot(x[route], y[route], marker='o', markersize=10,
             markerfacecolor="white", markeredgecolor="black", linewidth=1.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    # 输入数据
    dataset = np.loadtxt('input.txt')  # 数据中，每一列的含义分别为[序号，x坐标，y坐标]
    x = dataset[:, 1]  # x坐标
    y = dataset[:, 2]  # y坐标
    vertexes = dataset[:, 1: 3]  # 提取各个城市的xy坐标
    N = np.size(dataset, 0)
    h = np.array(pdist(vertexes))
    dist = np.array(squareform(h))  # 成本矩阵
    # 参数初始化
    MAXGEN = 50  # 外层最大迭代次数
    M = 100  # 最多进行M次邻域操作
    n = 3  # 邻域数目
    init_route, init_len = construct_route(dist)  # 贪婪构造初始解
    print('初始路线总距离为', init_len)
    cur_route = init_route
    best_route = cur_route
    best_len = route_length(cur_route, dist)
    BestL = np.zeros((MAXGEN, 1))  # 记录每次迭代过程中全局最优个体的总距离
    # 主循环
    gen = 0  # 外层计数器

    while gen < MAXGEN:
        k = 1
        while True:
            if k == 1:
                cur_route = shaking(cur_route, dist, k)[0]
                swap_route, swap_len = swap_neighbor(cur_route, dist, M)
                cur_len = swap_len
                if cur_len < best_len:
                    cur_route = swap_route
                    best_len = cur_len
                    best_route = swap_route
                    k = 0
            elif k == 2:
                cur_route = shaking(cur_route, dist, k)[0]
                reversion_route, reversion_len = reversion_neighbor(cur_route, dist, M)
                cur_len = reversion_len
                if cur_len < best_len:
                    cur_route = reversion_route
                    best_len = cur_len
                    best_route = reversion_route
                    k = 0
            elif k == 3:
                cur_route = shaking(cur_route, dist, k)[0]
                insertion_route, insertion_len = insertion_neighbor(cur_route, dist, M)
                cur_len = insertion_len
                if cur_len < best_len:
                    cur_route = insertion_route
                    best_len = cur_len
                    best_route = insertion_route
                    k = 0
            else:
                break
            k = k + 1
        # print(cur_len, best_len, "--488--")
        print('第', gen, '代最优路线总距离为', best_len)
        BestL[gen, 0] = best_len
        # 计数器加1
        gen = gen + 1

    plt.figure(1)
    plt.plot(BestL, linewidth=1)
    plt.title('优化过程')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # 画出全局最优路线图
    plot_route(best_route, x, y)