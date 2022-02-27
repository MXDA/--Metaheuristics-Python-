import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import math
import copy

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
            left = np.sum(part2[0: i])   # 在part1中，第i个旅行商访问城市的序号，即从start出发前往的下一个城市在part1中的序号
            right = np.sum(part2[0: i]) + part2[i] + 1  # 在part1中，第i个旅行商访问城市的序号，即返回至start的前一个城市在part1中的序号
            route = np.insert(part1[left: right], 0, start)  # 将start添加到这条路线的首末位置
            route = np.append(route, start)
        RP[i] = route
    return RP


def travel_distance(RP, dist):
    m = np.size(RP, 0)  # 旅行商数目
    everyTD = np.zeros(m)  # 初始化每个旅行商的行走距离
    for i in range(m):
        route = RP[i]  # 每个旅行商的行走路线
        everyTD[i] = route_length(route, dist)

    sumTD = sum(everyTD)  # 所有旅行商的行走总距离
    maxETD = max(everyTD)  # everyTD中的最大值
    return sumTD, everyTD, maxETD



def route_length(route, dist):
    n = np.size(route)  # 这条路线所经过城市的数目，包含起点和终点城市
    len = 0
    for k in range(n - 1):
        i = route[k]
        j = route[k + 1]
        len = len + dist[i, j]
    return len

def cross(individual1, individual2, n):
    cities_ind1 = individual1[0: n - 1]  # 灰狼个体1的中城市序列
    cities_ind2 = individual2[0: n - 1]  # 灰狼个体2的中城市序列
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

    individual1[0: n - 1] = cities_ind1
    individual2[0: n - 1] = cities_ind2
    return individual1, individual2
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

dataset = np.loadtxt('intest.txt')
x = np.array(dataset[:, 1])
y = np.array(dataset[:, 2])
vertexes = np.array(dataset[:, 1:])
n = np.size(dataset, 0)
m = 2
start = 0
h = np.array(pdist(vertexes))
dist = np.array(squareform(h))
print(dist)
a = encode(n, m, start)
print(a)
b = encode(n, m, start)
print(b)
c = decode(a, n, m, start)
print(c)
t = travel_distance(c, dist)[2]
print(t)