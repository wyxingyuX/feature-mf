# -*- coding:utf-8 -*-
import numpy as np
from sklearn import *

# b=np.arange(50).reshape(5,10)
# print preprocessing.normalize(b,norm="l1")

# a=np.random.uniform(0,1,size=10).reshape(5,2)
# m,n=a.shape
# print m,n

a=np.array([[1,2],[3,4]])
print a
print np.transpose(a)
b=np.array([[2,2],[1,1]])
print b

print np.where(b>0,0,1)

print np.linalg.inv(a)

print np.linalg.norm(a,ord="fro")

# c=np.array([[4,3],[2,1]])
#
# print np.dot(np.dot(a,b),c)
d=np.array([1,2,3,4,5])
print d
print np.sum(d)