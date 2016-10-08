# -*- coding:utf-8 -*-
# Paper: Multi-View Concept Learning for Data Representation
import numpy as np


def updateU(X, U, V):
    X_Vt = np.dot(X, V.T)
    U_V_Vt = np.dot(np.dot(U, V), V.T)
    m, kk = U.shape()
    for i in range(m):
        for k in range(kk):
            U[i, k] = 1.0 * U[i, k] * X_Vt[i, k] / U_V_Vt[i, k]
    return U


def getA(beta, P, Vl, Da, Wp):
    A = np.dot(P, Vl) + beta * (np.dot(Vl, (Da + Wp).T))
    return A


def getB(r, Ql):
    B = r - Ql
    return B


def getC(beta, Dp, Wa, Vl):
    C = beta * (np.dot(Vl, (Dp + Wa).T))
    return C


def getP(views_U):
    m, k = views_U[0].shape
    P = np.zeros((k, k))
    for i in range(len(views_U)):
        U = views_U[i]
        P += np.dot(U.T, U)
    return P


def getQl_Qu(views_X, Nl, views_U):
    m, k = views_U[0].shape
    m, N = views_X[0].shape
    Ql = np.zeros((k, Nl))
    Qu = np.zeros((k, N - Nl))
    for i in range(len(views_U)):
        U = views_U[i]
        X = views_X[i]
        Xl = X[:, 0:Nl]
        Xu = X[:, Nl:N]
        Ql += np.dot(U.T, Xl)
        Qu += np.dot(U.T, Xu)

    return Ql, Qu


def updateV(beta, r, V, Nl, P, Ql, Qu, Da, Wa, Dp, Wp):
    Vl = V[:, 0:Nl]
    kk, n = V.shape

    A = getA(beta, P, Vl, Da, Wp)
    B = getB(r, Ql)
    C = getC(beta, Dp, Wa, Vl)

    P_Vu = np.dot(P, V[:, Nl:n])
    for k in range(len(kk)):
        for j in range(len(n)):
            if j < Nl:
                tmp = (-B[k][j] + (B[k][j] * B[k][j] + 4 * A[k][j] * C[k][j]) ** 0.5) * V[k][j] / (2.0 * A[k][j])
                V[k][j] = min(1, tmp)
            else:
                u_j = j - Nl
                tt = (r - Qu[k, u_j])
                tmp = (-tt + abs(tt)) * V[k][j] / (2.0 * P_Vu[k, u_j])
                V[k][j] = min(1, tmp)
    return V


# Xmn is n items have m features, [Xl,Xu] Xl has label item,Xu don't have label item. y is n items label, -1 is no label
def get_La_Lp_Wa_Wp_Da_Dp(X, y):
    m, n = X.shape
    label_counts_dic = {}
    total_label_count = 0
    for i in range(len(y)):
        label = y[i]
        if label != -1:
            total_label_count += 1
            if label_counts_dic.has_key(label):
                label_counts_dic[label] += 1
            else:
                label_counts_dic[label] = 1

    Nl = total_label_count
    Nl_1 = 1.0 / Nl

    Wa = np.zeros((Nl, Nl))
    Wp = np.zeros((Nl, Nl))

    for j in range(Nl):
        cj = y[j]
        for i in range(j):
            ci = y[i]
            if ci == cj:
                Wa[i][j] = 1.0 / label_counts_dic[ci] - Nl_1
                Wa[j][i] = Wa[i][j]

                Wp[i][j] = Nl_1
                Wp[j][i] = Wp[i][j]
            else:
                Wa[i][j] = 0
                Wa[j][i] = Wa[i][j]

                Wp[i][j] = 0
                Wp[j][i] = Wp[i][j]

    Da = np.zeros((Nl, Nl))
    Dp = np.zeros((Nl, Nl))
    for i in range(Nl):
        for j in range(Nl):
            Da[i][i] += Wa[i][j]
            Dp[i][i] += Wp[i][j]
    La = Da - Wa
    Lp = Dp - Wp

    return La, Lp, Wa, Wp, Da, Dp


# def L(views_X,views_U,V,):
#     return L


def mcl(views_X, y, Nl, beta, r, nn, k):
    # Randomly initialize (Uik)v>=0, 1>=Vkj>=o, any j,k,v
    views_U = []
    V = np.random.uniform(0, 1, size=nn * k).reshape(nn, k)
    for i in range(len(views_X)):
        m, n = views_X[i].shape
        U = np.random.uniform(0, 1, size=m * k).reshape(m, k)
        views_U.append(U)

    itr = 0
    max_itr = 10
    while itr < max_itr:
        for i in range(len(views_U)):
            X = views_X[i]
            U = views_U[i]
            updateU(X, U, V)

        La, Lp, Wa, Wp, Da, Dp = get_La_Lp_Wa_Wp_Da_Dp(views_X[0], y)
        Ql, Qu = getQl_Qu(views_X, Nl, views_U)
        P = getP(views_U)
        updateV(beta, r, V, Nl, P, Ql, Qu, Da, Wa, Dp, Wp)

    return views_U, V


if __name__ == "__main__":
    views_X=[]
    X1 = np.random.uniform(0, 10, size=100).reshape(10, 10)
    X2 = np.random.uniform(20, 30, size=100).reshape(10, 10)
    y=[1,1,1,1,2,2,2,2,-1,-1]
    Nl=8
    nn=10
    k=5
    beta=0.02
    r=0.02
    views_X.append(X1)
    views_X.append(X2)

    view_U,V=mcl(views_X, y, Nl, beta, r, nn, k)
    print V