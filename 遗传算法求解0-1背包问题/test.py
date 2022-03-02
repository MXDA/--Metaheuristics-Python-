import numpy as np
import matplotlib.pyplot as plt

'''
t = np.array([1, 2, 3, 4])
a = np.array([0,1,0,1])
b = a==1;
w = t[b];
c = np.argwhere(a == 1).ravel()
d = np.array([1, 3, 2, 4, 5])
[e, f] = d.sort()
print(b);
print(w);
print(w.sum())
print(c)
print(e, f)
'''

'''
a = np.array([1, 3, 2, 4, 6])
b = np.argsort(a)
c = np.argsort(a)[::-1]
print(b)
print(c)
'''
'''
a = np.round(np.random.rand(5)).astype(int)
print(type(a))
print(a)
flag = False
if not flag:
    print(1)
'''
'''
返回第一个大于等于r值的下标
'''



'''
def find_first_greater(c, r):
    for i in range(np.size(c)):
        if c[i] >= r:
            return i


a = np.array([0.1, 0.2, 0.3, 0.4])
print(find_first_greater(a, 0.25))


def sum():
    a = 1
    b = 2
    return np.array([a, b])


A = np.array([[1, 2, 3], [3, 2, 1]])
print(np.size(A, 0))
print(A[0, :])
print(A[1, :])
b = np.zeros((2, 3)).astype(int)
print(np.zeros((2, 3)).astype(int))
print(type(b))
t = sum()
print(t)
print(type(t))
a = 0.9
c = int(5 * a)
print(c)
a = np.array([1, 2, 3, 4])
b = 10
c = a / b
print(c)
d = np.zeros((3, 1)).astype(int)

print(d)

e = np.array([[0.1, 0.2, 0.3, 0.4]])
f = np.cumsum(e)
print(f)
r = np.random.rand();
print(r)
index = np.argwhere(r <= e)
print(index)

a = np.array([[3, 1, 2], [5, 2, 1], [7, 1, 1]])
print(a[[0, 2], :])
for i in range(0, 9 - 9 % 2, 2):
    print(i, end=' ')
print("")

print(np.random.randint(5))

index = 3
c = [0, 1, 0, 1, 1, 0]
print(c[0:3])
print(np.random.permutation(5))
print(max(1, 2))
print(min(1, 2))
a = np.array([1, 2, 3, 4, 5])
a[1 : 3] = a[3 : 1 : -1]
print(a)
A = [[1, 2, 3], [3, 2, 1], [5, 1, 2]]
B = [[5, 6, 7], [8, 9, 2]]
print()
a = np.array([[1], [3], [2], [5]])
print(np.argmax(a))
print(np.max(a))
print('第', str(3), '次迭代的全局最优解为：', str(4))
'''

a = [1, 0, 1, 1, 0, 1, 0]
b = np.argwhere(a == 0).ravel()
b = [idx for (idx, val) in enumerate(a) if val == 1]
print(b)
b = np.array(b)
print(type(b))

R = np.random.permutation(5)
print(R)
print(R[0])
print(R[1])
a = np.array([2.1, 2.1 ,3, 1, 4, 5])
b = np.argsort(a)
print(b)
a = np.array([1, 2, 9, 7, 3, 6])
print(a[1 : 3], np.flipud(a[1 : 3]))
print(a)
a[1 : 3] = np.flipud(a[1 : 3])
print(a)