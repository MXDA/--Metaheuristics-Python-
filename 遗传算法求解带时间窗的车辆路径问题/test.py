import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import math
import copy
import os


s = np.empty(3, dtype = object)
s[0] = np.array([x for x in range(3)])
print(s[0])
s[1] = np.array([5, 4])
s[2] = np.array([6])
a = np.array([]).astype(int)
a = np.append(a, s[0])
print(a)
t = np.array([x + 1 for x in range(1, 5)])
print(t)
y = np.array([x for x in range(2, 6)])
print(y)