# -*- coding:utf-8 -*-
import numpy as np
import sklearn.decomposition as decomposition

import myutil as myu
from sklearn.utils import shuffle

# b=np.arange(50).reshape(5,10)
# print preprocessing.normalize(b,norm="l1")

# a=np.random.uniform(0,1,size=10).reshape(5,2)
# m,n=a.shape
# print m,n

a = np.array([[1, 2], [3, 4]])
print a
print np.transpose(a)
print myu.matrix_norm_1_1(a)
print a.trace()

b = np.array([[2, 2], [1, 1]])
print b

print np.where(b > 0, 0, 1)

print np.linalg.inv(a)

print np.linalg.norm(a, ord="fro")

# c=np.array([[4,3],[2,1]])
#
# print np.dot(np.dot(a,b),c)
d = np.array([1, 2, 3, 4, 5])
print d.size
print np.sum(d)
decomposition.NMF

from scipy.sparse import coo_matrix, hstack

A = coo_matrix([[1, 2], [3, 4]])
B = coo_matrix([[5, 6], [7, 8]])
print hstack([A, B]).todense()

tt = range(5)
print tt

aa = [1, 7, 3, 2, 6]
aa.sort()
print aa

cc = ["1", "2"]
tt = ["3", "4"]
print np.array([int(i) for i in cc]) + np.array([int(j) for j in tt])

ee = np.array([4, 4, 2, 2])

print type(ee), 2 * ee

a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
b = np.array(a)
print b.trace()
print (a + b) / 2

from sklearn.preprocessing import normalize

a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print normalize(a)

print np.random.rand(10, 5)
