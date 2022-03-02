import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
返回第一个大于等于r值的下标
'''


def find_first_greater(c, r):
    for i in range(np.size(c)):
        if c[i] >= r:
            return i


'''
判断一个个体是否满足背包的载重量约束，1表示满足，0表示不满足
输入Individual：          个体
输入w：                   各个物品的质量
输入cap：                 背包的载重量
输出flag：                表示一个个体是否满足背包的载重量约束，1表示满足，0表示不满足
'''


def judge_individual(Individual, w, cap):
    pack_item = Individual == 1  # 判断第i个位置上的物品是否装包，1表示装包，0表示未装包
    w_pack = w[pack_item]  # 找出装进背包中物品的质量
    total_w = w_pack.sum()  # 计算装包物品的总质量
    return total_w <= cap  # 如果装包物品的总质量小于等于背包的载重量约束，则为1，否则为0


'''
对违反约束的个体进行修复
输入Individual：          个体
输入w：                   各个物品的质量
输入p：                   各个物品的价值
输入cap：                 背包的载重量
输出Individual：          修复后的个体
'''


def repair_individual(Individual, w, p, cap):
    # 判断一个个体是否满足背包的载重量约束，1表示满足，0表示不满足
    flag = judge_individual(Individual, w, cap)
    # 只有不满足约束的个体才进行修复
    if flag == 0:
        # 初步修复
        pack_item = np.array([idx for (idx, val) in enumerate(Individual) if val == 1])  # 找出装进背包中物品的序号
        num_pack = np.size(pack_item)  # 装进背包中物品的总数目
        w_pack = w[pack_item]  # 找出装进背包中物品的质量
        total_w = w_pack.sum()  # 计算装包物品的总质量
        p_pack = p[pack_item]  # 找出装进背包中物品的价值
        ratio_pack = p_pack / w_pack  # 计算装进背包中物品的性价比=价值/质量
        rps_index = np.argsort(ratio_pack)  # 将已经装进包中的物品按照性价比（性价比=价值/质量）由低到高进行排序
        # 按照rps_index顺序，依次将物品从背包中移除
        for i in range(num_pack):
            remove_item = pack_item[rps_index[i]]  # 被移除的物品的序号
            # 如果移除该物品后满足背包的载重量约束，则将该物品对应的基因位改为0，然后终止循环
            if total_w - w_pack[rps_index[i]] <= cap:
                total_w = total_w - w_pack[rps_index[i]]  # 装包中物品总质量减少
                Individual[remove_item] = 0  # 将该物品对应的基因位改为0
                break
            else:
                # 如果移除该物品后依然不满足背包的载重量约束，则也要将该物品对应的基因位改为0，然后继续移除其它物品
                total_w = total_w - w_pack[rps_index[i]]
                Individual[remove_item] = 0

        # 进一步修复
        unpack_item = np.array([idx for (idx, val) in enumerate(Individual) if val == 0])  # 找出此时未装进背包中物品的序号
        num_unpack = np.size(unpack_item)  # 此时未装进背包中物品的总数目
        w_unpack = w[unpack_item]  # 找出此时未装进背包中物品的质量
        p_unpack = p[unpack_item]  # 找出此时未装进背包中物品的价值
        ratio_unpack = p_unpack / w_unpack  # 计算此时未装进背包中物品的性价比=价值/质量
        rups_index = np.argsort(ratio_unpack)[::-1]  # 将此时未装进包中的物品按照性价比（性价比=价值/质量）由高到低进行排序
        # 按照rups_index顺序，依次将物品装包
        for j in range(num_unpack):
            pack_wait = unpack_item[rups_index[j]]  # 待装包物品编号
            # 如果装包该物品后满足背包的载重量约束，则将该物品对应的基因位改为1，然后继续装包其它物品
            if total_w + w_unpack[rups_index[j]] <= cap:
                total_w = total_w + w_unpack[rups_index[j]]  # 装包中物品总质量增加
                Individual[pack_wait] = 1  # 将该物品对应的基因位改为1
            else:
                # 如果装包该物品后不满足背包的载重量约束，则终止循环
                break
    return Individual


'''
编码，生成满足约束的个体
输入n：                   物品数目
输入w：                   各个物品的质量
输入p：                   各个物品的价值
输入cap：                 背包的载重量
输出Individual：          满足背包载重量约束的个体
'''


def encode(n, w, p, cap):
    Individual = np.round(np.random.rand(n)).astype(int)  # 随机生成n个数字（每个数字是0或1）
    flag = judge_individual(Individual, w, cap)  # 判断Individual是否满足背包的载重量约束，1表示满足，0表示不满足
    # 如果flag为0，则需要修复个体Individual。否则，不需要修复
    if not flag:
        Individual = repair_individual(Individual, w, p, cap)
    return Individual


'''
调整种群染色体，将不满足载重量约束的染色体进行调整
输入Chrom：               种群
输入w：                   各个物品的质量
输入p：                   各个物品的价值
输入cap：                 背包载重量
输出Chrom：               调整后的染色体，全部满足载重量约束
'''


def adjustChrom(Chrom, w, p, cap):
    NIND = np.size(Chrom, 0)  # NIND种群大小
    for i in range(NIND):
        Individual = Chrom[i, :]  # 第i个个体
        flag = judge_individual(Individual, w, cap)  # 判断random_Individual是否满足背包的载重量约束，1表示满足，0表示不满足
        # 如果flag为0，则需要修复个体Individual。否则，不需要修复
        if not flag:
            Individual = repair_individual(Individual, w, p, cap)  # 修复个体Individual
            Chrom[i, :] = Individual  # 更新第i个个体
    return Chrom


'''
初始化种群
输入NIND：                种群大小
输入n：                   物品数目
输入w：                   各个物品的质量
输入p：                   各个物品的价值
输入cap：                 背包的载重量
输出Chrom：               初始种群
'''


def InitPop(NIND, n, w, p, cap):
    Chrom = np.zeros((NIND, n)).astype(int)  # 用于存储种群
    for i in range(NIND):
        Chrom[i, :] = encode(n, w, p, cap)  # 编码，生成满足约束的个体
    return Chrom


'''
计算单个染色体的装包物品总价值和总重量
输入n：                      物品数目
输入Individual：             个体
输入p：                      各个物品价值
输入w：                      各个物品质量
输出sumP：                   该个体的装包物品总价值
输出sumW：                   该个体的装包物品总重量
'''


def Individual_P_W(n, Individual, p, w):
    sumP = 0
    sumW = 0
    for i in range(n):
        # 如果为1，则表示物品被装包
        if Individual[i] == 1:
            sumP = sumP + p[i]
            sumW = sumW + w[i]
    return [sumP, sumW]


'''
计算种群中每个染色体的物品总价值
输入Chrom：               种群
输入p：                   各个物品的价值
输入w：                   各个物品的质量
输出Obj：                 种群中每个个体的物品总价值
'''


def Obj_Fun(Chrom, p, w):
    NIND = np.size(Chrom, 0)  # 种群大小
    n = np.size(Chrom, 1)  # 物品数目
    Obj = np.zeros((NIND, 1)).astype(int)

    for i in range(NIND):
        Obj[i, 0] = Individual_P_W(n, Chrom[i, :], p, w)[0]
    return Obj


'''
选择操作
输入Chrom：               种群
输入FitnV：               适应度值
输入GGAP：                代沟
输出SelCh：               被选择的个体
'''


def Select(Chrom, FitnV, GGAP):
    NIND = np.size(Chrom, 0)  # 种群数目
    Nsel = int(NIND * GGAP)
    total_FitnV = FitnV.sum()  # 所有个体的适应度之和
    select_p = FitnV / total_FitnV  # 计算每个个体被选中的概率
    select_index = np.zeros((Nsel, 1)).astype(int)  # 储存被选中的个体序号
    c = np.cumsum(select_p)  # 对select_p进行累加操作
    for i in range(Nsel):
        r = np.random.rand()  # 0~1之间的随机数
        index = find_first_greater(c, r)  # 每次被选择出的个体序号
        select_index[i, 0] = index
    Selch = Chrom[select_index.ravel(), :]  # 被选中的个体
    return Selch


'''
交叉操作
输入SelCh：               被选择的个体
输入Pc：                  交叉概率
输出SelCh：               交叉后的个体
'''


def Crossover(SelCh, Pc):
    Nels = np.size(SelCh, 0)
    n = np.size(SelCh, 1)  # n为染色体长度
    for i in range(0, (Nels - Nels % 2), 2):
        if Pc >= np.random.rand():  # 交叉概率Pc
            cross_pos = np.random.randint(n)  # 随机生成一个1~N之间的交叉位置
            cross_Selch1 = SelCh[i, :]  # 第i个进行交叉操作的个体
            cross_Selch2 = SelCh[i + 1, :]  # 第i+1个进行交叉操作的个体

            cross_part1 = cross_Selch1[0: cross_pos]  # 第i个进行交叉操作个体的交叉片段
            cross_part2 = cross_Selch2[0: cross_pos]  # 第i+1个进行交叉操作个体的交叉片段

            cross_Selch1[0: cross_pos] = cross_part2  # 用第i+1个个体的交叉片段替换掉第i个个体交叉片段
            cross_Selch2[0: cross_pos] = cross_part1  # 用第i个个体的交叉片段替换掉第i+1个个体交叉片段

            SelCh[i, :] = cross_Selch1  # 更新第i个个体
            SelCh[i + 1, :] = cross_Selch2  # 更新第i+1个个体
    return SelCh


'''
变异操作
输入SelCh：               被选择的个体
输入Pm：                  变异概率
输出SelCh：               变异后的个体
'''


def Mutate(SelCh, Pm):
    Nsel = np.size(SelCh, 0)
    n = np.size(SelCh, 1)  # n为染色体长度
    for i in range(Nsel):
        if Pm >= np.random.rand():
            R = np.random.permutation(n)  # 随机生成0~n-1的随机排列
            # print(n, R)
            pos1 = R[0]  # 第1个变异位置
            pos2 = R[1]  # 第2个变异位置

            left = min(pos1, pos2)  # 更小的那个值作为变异起点
            right = max(pos1, pos2)  # 更大的那个值作为变异终点

            SelCh[i, left: right] = np.flipud(SelCh[i, left: right])  # 更新第i个进行变异操作的个体
    return SelCh


'''
重插入子代的新种群
输入Chrom：               父代种群
输入SelCh：               子代种群
输入Obj：                 父代适应度
输出Chrom：               重组后得到的新种群
'''


def Reins(Chrom, SelCh, Obj):
    NIND = np.size(Chrom, 0)
    Nsel = np.size(SelCh, 0)
    index = np.argsort(Obj.ravel()).ravel()[::-1]
    Chrom = np.vstack((Chrom[index[0: NIND - Nsel], :], SelCh))
    return Chrom


if __name__ == '__main__':
    # 创建数据
    # 各个物品的质量，单位kg
    w = np.array([80, 82, 85, 70, 72, 70, 82, 75, 78, 45, 49, 76, 45, 35, 94, 49, 76, 79, 84, 74, 76, 63, \
                  35, 26, 52, 12, 56, 78, 16, 52, 16, 42, 18, 46, 39, 80, 41, 41, 16, 35, 70, 72, 70, 66, 50, 55, 25,
                  50, 55,
                  40])
    # 各个物品的价值，单位元
    p = np.array([200, 208, 198, 192, 180, 180, 168, 176, 182, 168, 187, 138, 184, 154, 168, 175, 198, \
                  184, 158, 148, 174, 135, 126, 156, 123, 145, 164, 145, 134, 164, 134, 174, 102, 149, 134, \
                  156, 172, 164, 101, 154, 192, 180, 180, 165, 162, 160, 158, 155, 130, 125])
    cap = 1000  # 每个背包的载重量为1000kg
    n = np.size(p)  # 物品个数
    # 参数设置
    NIND = 500  # 种群大小
    MAXGEN = 500  # 迭代次数
    Pc = 0.9  # 交叉概率
    Pm = 0.08  # 变异概率
    GGAP = 0.9  # 代沟
    # 初始化种群
    Chrom = InitPop(NIND, n, w, p, cap)
    # 优化
    gen = 0
    bestIndividual = Chrom[0, :]  # 初始将初始种群中一个个体赋值给全局最优个体
    bestObj = Individual_P_W(n, bestIndividual, p, w)[0]  # 计算初始bestIndividual的物品总价值
    BestObj = np.zeros((MAXGEN, 1))  # 记录每次迭代过程中的最优适应度值
    while gen < MAXGEN:
        # 计算适应度
        Obj = Obj_Fun(Chrom, p, w)  # 计算每个染色体的物品总价值
        FitnV = Obj  # 适应度值=目标函数值=物品总价值
        #   选择
        SelCh = Select(Chrom, FitnV, GGAP)
        #   交叉操作
        SelCh = Crossover(SelCh, Pc)
        #   变异
        SelCh = Mutate(SelCh, Pm)
        #   重插入子代的新种群
        Chrom = Reins(Chrom, SelCh, Obj)
        #   将种群中不满足载重量约束的个体进行约束处理
        Chrom = adjustChrom(Chrom, w, p, cap)
        #   记录每次迭代过程中最优目标函数值
        cur_bestObj = np.max(Obj)
        cur_bestIndex = np.argmax(Obj)  # 在当前迭代中最优目标函数值以及对应个体的编号
        cur_bestIndividual = Chrom[cur_bestIndex, :]
        # 如果当前迭代中最优目标函数值大于等于全局最优目标函数值，则进行更新
        if cur_bestObj >= bestObj:
            bestObj = cur_bestObj
            bestIndividual = cur_bestIndividual
        BestObj[gen, 0] = bestObj  # 记录每次迭代过程中最优目标函数值
        #   打印每次迭代过程中的全局最优解
        print('第', str(gen), '次迭代的全局最优解为：', str(bestObj))
        #   更新迭代次数
        gen = gen + 1
    #   画出迭代过程图
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'FangSong'
    plt.rcParams['figure.dpi'] = 75
    sns.set_theme(style='darkgrid')
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数值（物品总价值）')
    sns.lineplot(data=BestObj)
    plt.show()
    #   最终装进包中的物品序号
    pack_item = np.argwhere(bestIndividual == 1)
    [bestP, bestW] = Individual_P_W(n, bestIndividual, p, w)
    print(bestP)
    print(bestW)
