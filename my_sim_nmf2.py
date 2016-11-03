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


def compute_loss(views_X, views_Laplace, views_lamada, views_gamma, alpha1, alpha2, beta, views_U, views_V, V_centroid):
    loss = np.linalg.norm(V_centroid) * beta * 0.5
    for i in range(len(views_X)):
        X = views_X[i]
        U = views_U[i]
        V = views_V[i]
        Laplace = views_Laplace[i]
        loss += 0.5 * np.linalg.norm(X - np.dot(U, V)) \
                + views_lamada[i] * np.dot(np.dot(V, Laplace), V.T).trace() \
                + 0.5 * alpha1 * np.linalg.norm(U) \
                + 0.5 * alpha2 + np.linalg.norm(V) \
                + 0.5 * views_gamma[i] * np.linalg.norm(V - V_centroid)
    return loss


def gen_new_U_V_Vcentroid(views_U, views_V, V_centroid, gradU, gradV, gradV_centroid, r):
    Vcentroid_subGrad = V_centroid - r * gradV_centroid
    viewU_subGrad = []
    viewV_subGrad = []
    for i in range(len(views_U)):
        viewU_subGrad.append(views_U[i] - r * gradU[i])
        viewV_subGrad.append(views_V[i] - r * gradV[i])
    return viewU_subGrad, viewV_subGrad, Vcentroid_subGrad


# Xmn ~ Umk x Vkn
def sim_nmf(views_X, views_lamada, views_gamma, alpha1=0.01, alpha2=0.01, beta=0.01, k=20, max_itr=20, min_error=0.5,
            is_nomorlize_X=True):
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
    views_V = []
    views_Laplace = gen_sim_laplace(views_X)
    V_centroid = np.zeros((k, n))

    for i in range(len(views_X)):
        X = views_X[i].T  # X=X.T , X shape now is nm, row is sample
        nmf_model = NMF(n_components=k)
        Vi = nmf_model.fit_transform(X)
        Ui = nmf_model.components_

        views_U.append(Ui.T)
        views_V.append(Vi.T)
        V_centroid = V_centroid + Vi.T
    if len(views_X) != 0:
        V_centroid /= 1.0 * len(views_X)

    Lt0 = 1000000
    Lt1 = 1
    itr = 0
    while itr < max_itr and (Lt0 - Lt1) > min_error:
        print "itr:", str(itr)
        gradV_centroid = beta * V_centroid
        for i in range(len(views_X)):
            gradV_centroid += views_gamma[i] * (V_centroid - views_V[i])

        gradV = []
        for i in range(len(views_X)):
            X = views_X[i]
            U = views_U[i]
            V = views_V[i]
            Laplace = views_Laplace[i]
            # print len(views_X),i
            gradV_i = alpha2 * V + np.dot(np.dot(U.T, U), V) - np.dot(U.T, X) + views_lamada[i] * np.dot(V, (
                Laplace + Laplace.T)) + \
                      views_gamma[i] * (V - V_centroid)
            gradV.append(gradV_i)

        gradU = []
        for i in range(len(views_X)):
            X = views_X[i]
            U = views_U[i]
            V = views_V[i]

            gradU_i = alpha1 * U + (np.dot(np.dot(U, V), V.T) - np.dot(X, V.T))
            gradU.append(gradU_i)

        r = 1.0  # learning rate
        early_stop = False
        loss = compute_loss(views_X, views_Laplace, views_lamada, views_gamma, alpha1, alpha2, beta, views_U, views_V, \
                            V_centroid)
        viewU_subGrad, viewsV_subGrad, Vcentroid_subGrad = gen_new_U_V_Vcentroid(views_U, views_V, V_centroid, gradU, \
                                                                                 gradV, gradV_centroid, r)
        subtGradLoss = compute_loss(views_X, views_Laplace, views_lamada,views_gamma, alpha1, alpha2, beta, viewU_subGrad, \
                                    viewsV_subGrad, Vcentroid_subGrad)
        while (loss - subtGradLoss) < 0:
            r /= 2.0  # search for the maximal step size
            viewU_subGrad, viewsV_subGrad, Vcentroid_subGrad = gen_new_U_V_Vcentroid(views_U, views_V, V_centroid,
                                                                                     gradU, gradV, gradV_centroid, r)
            subtGradLoss = compute_loss(views_X, views_Laplace,views_lamada, views_gamma, alpha1, alpha2, beta, viewU_subGrad,
                                        viewsV_subGrad, Vcentroid_subGrad)
            if r <= 0.0000001:
                early_stop = True
                break
            print "r:", str(r), str(loss), str(subtGradLoss)
        if not early_stop:
            views_U = viewU_subGrad
            views_V = viewsV_subGrad
            V_centroid = Vcentroid_subGrad
        Lt0 = loss
        Lt1 = subtGradLoss

        itr += 1
    end_t = time.clock()
    print "itr:", str(itr), "delt:", str(Lt0 - Lt1), " cost time:", str(end_t - start_t)
    return V_centroid, views_U, views_V


if __name__ == "__main__":
    print "hello"
