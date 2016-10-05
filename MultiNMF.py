# -*- coding:utf-8 -*-
# paper：Multi-View Clustering via Joint Nonnegative Matrix Factorization
import numpy as np
from sklearn import *


def updateU(X, U, V, V_consensus, view_weight):
    m, k = U.shape
    n, k = V.shape

    X_V = np.dot(X, V)
    U_Vt_V = np.dot(np.dot(U, np.transpose(V)), V)

    V_Vc = np.zeros((1, k))
    for kk in range(k):
        result = 0
        for j in range(n):
            result += V[j, kk] * V_consensus[j, kk]
        V_Vc[0, kk] = result

    V_V = np.zeros((1, k))
    for kk in range(k):
        result = 0
        for j in range(n):
            result += V[j, kk] * V[j, kk]
        V_V[0, kk] = result

    U_VV = np.zeros((1, k))
    for kk in range(k):
        result = 0
        for l in range(m):
            result += U[l, kk]
        U_VV[0, kk] = result * V_V[0, kk]

    for i in range(m):
        for kk in range(k):
            U[i, kk] = 1.0 * U[i, kk] * (X_V[i, kk] + view_weight * V_Vc[0, kk]) / (
                U_Vt_V[i, kk] + view_weight * U_VV[0, kk])
    return U


def updateQ(U, Q):
    m, k = U.shape
    k, k = Q.shape
    for i in range(k):
        sum = 0
        for mm in range(m):
            sum += U[mm, i]
        Q[i, i] = sum
    return Q


def updateV(X, U, V, V_consensus, view_weight):
    Xt_U = np.dot(np.transpose(X), U)
    V_Ut_U = np.dot(np.dot(V, np.transpose(U)), U)
    V = np.where(V > -1, V * (Xt_U + view_weight * V_consensus) / (V_Ut_U + view_weight * V), V)
    return V


def computL(X, U, V, V_consensus, Q, view_weight):
    L_factor = np.linalg.norm(X - np.dot(U, np.transpose(V)), ord="fro")
    L_consensus = view_weight * np.linalg.norm(np.dot(V, Q) - V_consensus, ord="fro")
    return (L_factor + L_consensus), L_factor, L_consensus


def insert(arry, item, i):
    if i > (len(arry) - 1):
        arry.append(item)
    else:
        arry[i] = item


def multi_view_nmf(views_X, views_weight, nn, k):
    views_U = []
    views_V = []
    views_Q = []
    views_Cost = []
    V_consensus = np.random.uniform(0, 1, size=nn * k).reshape(nn, k)
    for i in range(len(views_X)):
        m, n = views_X[i].shape
        U = np.random.uniform(0, 1, size=m * k).reshape(m, k)
        V = np.random.uniform(0, 1, size=n * k).reshape(n, k)
        Q = np.zeros((k, k))
        updateQ(U, Q)
        views_U.append(U)
        views_V.append(V)
        views_Q.append(Q)
        views_Cost.append(0)

    sum_O_last = 0
    sum_O_cur = 10
    iter = 0
    # repeat,until Eq.3.6 converges (What show this Eq. converges? paper say after about 15 iter will convergence)
    while iter < 15:
        iter += 1
        sum_O_last = sum_O_cur
        for i in range(len(views_X)):
            X = preprocessing.normalize(views_X[i], norm="l1")
            m, n = X.shape
            if nn != n:
                print "Error: n！=nn"
                return
            U = views_U[i]
            V = views_V[i]
            Q = views_Q[i]

            O_last = 0
            O_cur = 1
            # repeat, unitl Eq. 3.7 converge
            while (O_cur - O_last) > 0.1:
                O_last = O_cur
                updateU(X, U, V, V_consensus, views_weight[i])

                # normalize U,V
                updateQ(U, Q)
                U = np.dot(U, np.linalg.inv(Q))
                V = np.dot(V, Q)

                updateV(X, U, V, V_consensus, views_weight[i])

                O_cur, O_factorize, O_consensus = computL(X, U, V, V_consensus, Q, views_weight[i])
            views_U[i] = U
            views_V[i] = V
            views_Q[i] = Q
            views_Cost[i] = O_cur
        # Fixing U and V , update V_consensus by Eq. 3.11
        weights = np.sum(views_weight)
        for nv in range(len(views_weight)):
            V_consensus += views_weight[nv] * np.dot(views_V[nv], views_Q[nv])
        V_consensus = V_consensus * (1.0 / weights)
        sum_O_cur = np.sum(views_Cost)
    return views_U, views_V, V_consensus


if __name__ == "__main__":
    views_X = []
    X1 = np.random.uniform(0, 10, size=100).reshape(10, 10)
    X2 = np.random.uniform(20, 30, size=100).reshape(10, 10)
    views_X.append(X1)
    views_X.append(X2)

    views_weight = [0.1, 0.9]
    views_U, views_V, V_consensus = multi_view_nmf(views_X, views_weight, 10, 5)
    print V_consensus