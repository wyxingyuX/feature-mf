# -*- coding:utf-8 -*-
# Paper: Multi-View Concept Learning for Data Representation

import numpy as np


def mcl(views_X, a, b, r, nn, k):
    # Randomly initialize (Uik)v>=0, 1>=Vkj>=o, any j,k,v
    views_U = []
    V = np.random.uniform(0, 1, size=nn * k).reshape(nn, k)
    for i in range(len(views_X)):
        m,n=views_X[i].shape
        U=np.random.uniform(0,1,size=m*k).reshape(m,k)
        views_U.append(U)

    while True:
        pass

    return views_U, V


if __name__ == "__main__":
    print "hello"
