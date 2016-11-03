# -*- coding:utf-8 -*-
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.preprocessing import minmax_scale
import numpy as np
import time


# X colum vec is unit
def gen_sim_laplace(views_X):
    views_laplace = []
    for i in range(len(views_X)):
        X = views_X[i]
        n = X.shape[1]
        W = np.dot(X.T, X)

        # dialog must 0?
        nw, nw = W.shape
        for ii in range(nw):
            W[ii][ii] = 0

        D = np.zeros((n, n))
        for i in range(n):
            D[i][i] = sum(W[:, i])

        L = D - W
        views_laplace.append(L)
    return views_laplace


def compute_loss(views_X, views_Laplace, views_lamada, alpha, beta, views_U, V):
    loss = np.linalg.norm(V) * beta * 0.5
    for i in range(len(views_X)):
        X = views_X[i]
        U = views_U[i]
        Laplace = views_Laplace[i]
        loss += 0.5 * np.linalg.norm(X - np.dot(U, V)) \
                + views_lamada[i] * np.dot(np.dot(V, Laplace), V.T).trace() \
                + 0.5 * alpha * np.linalg.norm(U)
    return loss


def gen_new_U_V(views_U, V, gradU, gradV, r):
    V_subGrad = V - r * gradV
    viewU_subGrad = []
    for i in range(len(views_U)):
        viewU_subGrad.append(views_U[i] - r * gradU[i])
    return viewU_subGrad, V_subGrad


# Xmn ~ Umk x Vkn
def sim_nmf(views_X, views_lamada, alpha=0.01, beta=0.01, k=20, max_itr=20, min_error=0.000001, is_nomorlize_X=True):
    start_t = time.clock()
    n = views_X[0].shape[1]
    print n
    # Normalize X
    for i in range(len(views_X)):
        X = views_X[i].T  # X=X.T , X shape now is nm, row is sample
        if n != X.shape[0]:
            print "views sample num is not equal!"
            return
        X = minmax_scale(X)
        if is_nomorlize_X:
            X = normalize(X)
        views_X[i] = X.T  # transpose X , now column is sample
    # Use NMF init U,V
    views_U = []
    views_Laplace = gen_sim_laplace(views_X)
    V = np.zeros((k, n))

    for i in range(len(views_X)):
        X = views_X[i].T  # X=X.T , X shape now is nm, row is sample
        nmf_model = NMF(n_components=k)
        Vi = nmf_model.fit_transform(X)
        Ui = nmf_model.components_

        views_U.append(Ui.T)
        V = V + Vi.T
    if len(views_X) != 0:
        V /= 1.0 * len(views_X)

    Lt0 = 1000000
    Lt1 = 1
    itr = 0
    while itr < max_itr and (Lt0 - Lt1) > min_error:
        print "itr:", str(itr)
        gradV = beta * V
        for i in range(len(views_X)):
            X = views_X[i]
            U = views_U[i]
            Laplace = views_Laplace[i]
            # print len(views_X),i
            gradV += np.dot(np.dot(U.T, U), V) - np.dot(U.T, X) + \
                     views_lamada[i] * np.dot(V, Laplace + Laplace.T)

        gradU = []
        for i in range(len(views_X)):
            X = views_X[i]
            U = views_U[i]

            gradU_i = alpha * U + (np.dot(np.dot(U, V), V.T) - np.dot(X, V.T))
            gradU.append(gradU_i)

        r = 1.0  # learning rate
        early_stop = False
        loss = compute_loss(views_X, views_Laplace, views_lamada, alpha, beta, views_U, V)

        viewU_subGrad, V_subGrad = gen_new_U_V(views_U, V, gradU, gradV, r)
        subtGradLoss = compute_loss(views_X, views_Laplace, views_lamada, alpha, beta, viewU_subGrad, V_subGrad)
        while (loss - subtGradLoss) < 0:
            r /= 2.0  # search for the maximal step size
            viewU_subGrad, V_subGrad = gen_new_U_V(views_U, V, gradU, gradV, r)
            subtGradLoss = compute_loss(views_X, views_Laplace, views_lamada, alpha, beta, viewU_subGrad, V_subGrad)
            if r <= 0.0000001:
                early_stop = True
                break
            print "r:", str(r), str(loss), str(subtGradLoss)
        if not early_stop:
            views_U = viewU_subGrad
            V = V_subGrad
        Lt0 = loss
        Lt1 = subtGradLoss

        itr += 1
    end_t = time.clock()
    print "itr:", str(itr), "delt:", str(Lt0 - Lt1), " cost time:", str(end_t - start_t)
    return V, views_U


if __name__ == "__main__":
    print "hello"
    X1 = np.random.rand(5000, 200)
    print type(X1)
    X2 = np.random.rand(5000, 200)
    views_X = []
    views_X.append(X1)
    views_X.append(X2)

    V, views_U = sim_nmf(views_X, 0.001, 0.01, k=3)
    print V.T
