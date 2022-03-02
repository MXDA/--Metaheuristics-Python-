import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import math
import copy
import os
def Swap(Scurr):
    n = np.size(Scurr)
    seq = np.random.permutation(n)
    I = seq[0: 2]
    i1 = I[0]
    i2 = I[1]
    Snew = copy.deepcopy(Scurr)
    Snew[i1], Snew[i2] = Scurr[i2], Scurr[i1]
    return Snew
def Reversion(Scurr):
    n = np.size(Scurr)
    seq = np.random.permutation(n)
    I = seq[0: 2]
    i1 = I[0]
    i2 = I[1]
    Snew = copy.deepcopy(Scurr)
    print(i1, i2)
    Snew[i1: i2 + 1] = np.flipud(Scurr[i1: i2 + 1])
    return Snew
def Insertion(Scurr):
    n = np.size(Scurr)
    if n <= 1:
        return Scurr
    seq = np.random.permutation(n)
    I = seq[0: 2]
    i1 = I[0]
    i2 = I[1]
    print(i1, i2)
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
a = np.array([1, 3, 2, 4, 6])
b = Insertion(a)
print(a, np.size(a))
print(b, np.size(b))
