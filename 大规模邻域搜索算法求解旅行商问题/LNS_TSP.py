import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import math
import copy
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
破坏函数destroy根据从当前解中连续移除若干个城市
输入route：                   当前解，一条路线
输出Sdestroy：                移除removed中的城市后的route
输出removed：                 被移除的城市集合
'''


def destroy(route):
    N = np.size(route)  # 当前解中城市数目
    Lmin = 1  # 一条路径中所允许移除最小的城市数目
    Lmax = min(int(math.ceil(N / 2)), 25)  # 一条路径中所允许移除最大的城市数目
    visit = int(math.ceil(np.random.rand() * N)) - 1  # 从当前解中随机选出要被移除的城市

    L = Lmin + int(math.ceil((Lmax - Lmin) * np.random.rand()))  # 计算在该条路径上移除的城市的数目
    findv = np.argwhere(route == visit).ravel()[0]  # 找出visit在route中的位置
    vLN = findv  # visit左侧的城市个数
    vRN = N - findv  # visit右侧的城市个数

    # 如果vLN小
    if vLN <= vRN:
        if (vRN < L - 1) and (vLN < L - 1):
            nR = L - 1 - vLN + int(np.round(np.random.rand() * (vRN - L + 1 + vLN)))
            nL = L - 1 - nR  # visit左侧要移除城市的数目
        elif (vRN > L - 1) and (vLN > L - 1):
            nR = int(np.round(np.random.rand() * vLN))  # visit右侧要移除城市的数目
            nL = L - 1 - nR  # visit左侧要移除城市的数目
        else:
            nR = L - 1 - vLN + int(np.round(np.random.rand() * vLN))
            nL = L - 1 - nR  # visit左侧要移除城市的数目
    else:
        # 如果vRN小
        if (vLN < L - 1) and (vRN < L - 1):
            nL = L - 1 - vRN + int(np.round(np.random.rand() * (vLN - L + 1 + vRN)))
            nR = L - 1 - nL  # visit右侧要移除城市的数目
        elif (vLN > L - 1) and (vRN > L - 1):
            nL = int(np.round(np.random.rand() * vRN))  # visit左侧要移除城市的数目
            nR = L - 1 - nL  # visit右侧要移除城市的数目
        else:
            nL = L - 1 - vRN + int(np.round(np.random.rand() * vRN))
            nR = L - 1 - nL

    removed = route[findv - nL: findv + nR]  # 移除城市的集合，即包括visit在内的连续L个城市
    Sdestroy = route  # 复制route
    for x in removed:
        Sdestroy = np.delete(Sdestroy, np.where(Sdestroy == x)) # %将removed中的所有城市从route中移除
    return Sdestroy, removed


'''
将visit插回到插入成本最小的位置后的路线，同时还计算出插入到各个插入位置的插入成本
输入visit：               待插入的城市
输入dist：                距离矩阵
输入route：               被插入路径
输出new_route：           将visit插入到route最小插入成本位置后的解
输出up_delta：            将visit插入到route中各个插入位置后的插入成本从小到大排序后的结果
'''


def ins_route(visit, dist, route):
    lr = np.size(route)  # 当前路线城市数目
    rc0 = np.zeros((lr + 1, lr + 1)).astype(int)  # 记录插入城市后的路径
    delta0 = np.zeros((lr + 1, 1))  # 记录插入城市后的增量
    for i in range(lr + 1):
        if i == lr:
            rc = np.append(route, visit)
        elif i == 0:
            rc = np.insert(route, 0, visit)
        else:
            rc = np.insert(route, i - 1, visit)
        rc0[i, :] = rc  # 将合理路径存储到rc0，其中rc0与delta0对应
        dif = route_length(rc, dist) - route_length(route, dist)  # 计算成本增量
        delta0[i, 0] = dif

    up_delta = np.sort(delta0)  # 将插入成本从小到大排序
    ind = np.argmin(delta0)  # 计算最小插入成本所对应的序号
    new_route = rc0[ind, :]
    return new_route, up_delta


'''
修复函数repair依次将removed中的城市插回路径中
先计算removed中各个城市插回当前解中所产生最小增量，然后再从上述各个最小增量的城市中
找出一个(距离增量第2小-距离增量第1小)最大的城市插回，反复执行，直到全部插回
'''


def repair(removed, Sdestroy, dist):
    Srepair = Sdestroy
    # 反复插回removed中的城市，直到全部城市插回
    while np.size(removed) != 0:
        nr = np.size(removed)  # 移除集合中城市数目
        regret = np.zeros((nr, 1))  # 存储removed各城市插回最“好”插回路径后的遗憾值增量
        # 逐个计算removed中的城市插回当前解中各路径的目标函数值增
        for i in range(nr):
            visit = removed[i]  # 当前要插回的城市
            up_delta = ins_route(visit, dist, Srepair)[1]  # 将visit插回到插入成本最小的位置后的路线，同时还计算出插入到各个插入位置的插入成本
            del2 = up_delta[1] - up_delta[0]  # 计算第2小成本增量与第1小成本增量差值
            regret[i] = del2  # 更新当前城市插回最“好”插回路径后的遗憾值
        max_index = np.argmax(regret)  # 找出遗憾值最大的城市序号
        reinsert_city = removed[max_index]  # removed中准备插回的城市
        Srepair = ins_route(reinsert_city, dist, Srepair)[0]  # 将reinsert_city插回到Srepair
        removed = np.delete(removed, max_index)

    repair_length = route_length(Srepair, dist)  # 计算Srepair的总距离
    return Srepair, repair_length


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
    dataset = np.loadtxt('input.txt')  # 数据中，每一列的含义分别为[序号，x坐标，y坐标]
    x = np.array(dataset[:, 1])  # x坐标
    y = np.array(dataset[:, 2])  # y坐标
    vertexes = np.array(dataset[:, 1:])  # 提取各个城市的xy坐标
    h = np.array(pdist(vertexes))
    dist = np.array(squareform(h))  # 距离矩阵

    # 参数初始化
    MAXGEN = 500  # 最大迭代次数
    # 构造初始解
    Sinit, init_len = construct_route(dist)  # 贪婪构造初始解
    init_length = route_length(Sinit, dist)
    print('初始总路线长度 =  ', str(init_length))
    Scurr = Sinit
    curr_length = init_length
    Sbest = Sinit
    best_length = init_length

    gen = 0
    BestL = np.zeros((MAXGEN, 1))  # 记录每次迭代过程中全局最优个体的总距离
    while gen < MAXGEN:
        # “破坏”解
        Sdestroy, removed = destroy(Scurr)
        # “修复”解

        Srepair, repair_length = repair(removed, Sdestroy, dist)
        if repair_length < curr_length:
            Scurr = Srepair
            curr_length = repair_length
        if curr_length < best_length:
            Sbest = Scurr
            best_length = curr_length
        # 打印当前代全局最优解
        print('第', str(gen), '代最优路线总长度 = ', str(best_length))
        BestL[gen, 0] = best_length
        # 计数器加1
        gen = gen + 1
    print('搜索完成！ 最优路线总长度 =  ', str(best_length))
    plt.figure(1)
    plt.plot(BestL, linewidth=1)
    plt.title('优化过程')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plot_route(Sbest, x, y)
