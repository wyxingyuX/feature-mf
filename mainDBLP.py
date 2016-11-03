# -*- coding:utf-8 -*-
import time

import numpy as np
import scipy.io as sio
from sklearn import cross_validation
from sklearn.decomposition import NMF
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import *
from sklearn.cross_validation import train_test_split
from sklearn import svm

import myutil as myu


def link_sparse_vec(src_file, docid_mycode_dic, dest_file):
    src_f = open(src_file)
    dest_f = open(dest_file, "w")
    for line in src_f:
        elms = line.split()
        doc_id = elms[0]
        if docid_mycode_dic.has_key(doc_id):
            dest_f.write(docid_mycode_dic[doc_id])
            total_link = int(elms[1])
            for i in range(0, 2 * total_link, 2):
                link_id = elms[i + 2].strip()
                link_w = elms[i + 3].strip()
                if docid_mycode_dic.has_key(link_id):
                    dest_f.write("\t" + docid_mycode_dic[link_id] + ":" + link_w)
            dest_f.write("\n")
    src_f.close()
    dest_f.close()


def text_sparse_vec(src_file, docid_mycode_dic, word_mycode_dic, dest_file):
    src_f = open(src_file)
    dest_f = open(dest_file, "w")
    for line in src_f:
        elms = line.split()
        doc_id = elms[0]
        if docid_mycode_dic.has_key(doc_id):
            dest_f.write(docid_mycode_dic[doc_id])
            total_words = int(elms[1])
            for i in range(0, 2 * total_words, 2):
                word = elms[i + 2]
                word_freq = elms[i + 3]
                dest_f.write("\t" + word_mycode_dic[word] + ":" + word_freq)
            dest_f.write("\n")
    src_f.close()
    dest_f.close()


def my_docid_code(src_file, dest_file):
    src_f = open(src_file)
    dest_f = open(dest_file, "w")
    line_num = 0

    for line in src_f:
        if line_num == 0:
            line_num += 1
            continue
        elms = line.split()
        id = elms[1]
        dest_f.write(id + "\t" + str(line_num) + "\n")
        line_num += 1

    src_f.close()
    dest_f.close()


def word_mycode(src_file, dest_file):
    src_f = open(src_file)
    word_code_dic = {}
    for line in src_f:
        elms = line.split()
        id = elms[0]
        total_word = int(elms[1])
        for i in range(0, 2 * total_word, 2):
            word = elms[i + 2]
            if not word_code_dic.has_key(word):
                word_code_dic[word] = len(word_code_dic) + 1
    src_f.close()

    dest_f = open(dest_file, "w")
    for word, code in word_code_dic.items():
        dest_f.write(word + "\t" + str(code) + "\n")
    dest_f.close()


def read_key_code_dic(src_file, ignore_linenum=0, inverse=False):
    src_f = open(src_file)
    key_code_dic = {}
    line_num = 0
    for line in src_f:
        line_num += 1
        if line_num <= ignore_linenum:
            continue
        elms = line.split()
        k_idx = 1 if inverse else 0
        v_idx = 0 if inverse else 1
        key = elms[k_idx].strip()
        code = elms[v_idx].strip()
        if not key_code_dic.has_key(key):
            key_code_dic[key] = code
    src_f.close()
    return key_code_dic


def docidmycode_cate(src_file, docid_mycode_dic, dest_file):
    src_f = open(src_file)
    dest_f = open(dest_file, "w")

    line_num = 0
    for line in src_f:
        if line_num == 0:
            line_num += 1
            continue
        elms = line.split()
        cate = elms[0]
        docid = elms[1]
        cate_name = elms[2]
        dest_f.write(docid_mycode_dic[docid] + "\t" + cate + "\t" + cate_name + "\n")
    src_f.close()
    dest_f.close()


def cate_count(cate_key_file):
    f = open(cate_key_file)
    cate_count_dic = {}
    for line in f:
        elms = line.split()
        key = elms[0]
        cate = elms[1]
        if cate_count_dic.has_key(cate):
            cate_count_dic[cate] += 1
        else:
            cate_count_dic[cate] = 1
    f.close()
    for cate, count in cate_count_dic.items():
        print cate, count


def read_X_y(id_features_file, id_cate_file):
    docid_cate_dic = read_key_code_dic(id_cate_file, ignore_linenum=1, inverse=True)
    f = open(id_features_file)
    sparse_vecs = []
    y = []
    for line in f:
        elms = line.split()
        docid = elms[0]
        vec = elms[1:-1]
        if docid_cate_dic.has_key(docid):
            sparse_vecs.append(vec)
            y.append(float(docid_cate_dic[docid]))
    f.close()
    X = myu.transf_csr_matrix_vecs(sparse_vecs)
    return X, y


def dblp_text_zn():
    base_path = "F:\ExpData\work\DBLP"

    X, y = read_X_y(base_path + "\\doc_vec_text.txt", base_path + "\\answerkeys_indexed.list")
    print X.shape, len(y)

    clf=LogisticRegression()
    sc= cross_validation.cross_val_score(clf, X, y, cv=5)

    print "zn:", np.mean(sc)


if __name__ == "__main__":
    print "hello"
    base_path = "F:\ExpData\work\\"

    dblp_text_zn()

    # my_docid_code("F:\ExpData\work\DBLP\\answerkeys_indexed.list","F:\ExpData\work\my\DBLP\\docid_mycode.txt")
    # word_mycode("F:\ExpData\work\DBLP\\dblp.collection","F:\ExpData\work\my\DBLP\\dblp_word_mycode.txt")
    # docid_mycode_dic = read_key_code_dic(base_path + "my\DBLP\\docid_mycode.txt")
    # link_sparse_vec(base_path + "DBLP\\dblplink.collection", docid_mycode_dic,
    #                 base_path + "my\DBLP\\link_sparse_vec.txt")
    # word_mycode_dic = read_key_code_dic(base_path + "my\DBLP\\dblp_word_mycode.txt")
    # text_sparse_vec(base_path + "DBLP\\dblp.collection", docid_mycode_dic, word_mycode_dic,
    #                 base_path + "my\DBLP\\text_sparse_vec.txt")
    # docidmycode_cate(base_path+"DBLP\\answerkeys_indexed.list",docid_mycode_dic,base_path+"my\DBLP\\docidmycode_cate_index.txt")


    # cate_count(base_path+"my\DBLP\\docidmycode_cate_index.txt")
    #
    # X_link_origin, y_link = read_X_y(base_path + "my\DBLP\\link_sparse_vec.txt",
    #                           base_path + "my\DBLP\\docidmycode_cate_index.txt")
    # X_text_origin, y_text = read_X_y(base_path + "my\DBLP\\text_sparse_vec.txt",
    #                           base_path + "my\DBLP\\docidmycode_cate_index.txt")

    # print "start save mat"
    # sio.savemat(base_path + "my\DBLP\\dblp.mat", {'link': X_link_origin, "text": X_text_origin,"gnd":y_text})
    # print "end save mat"


    # print "start", time.localtime()
    # nmf_model1=NMF(n_components=50)
    # X_link=nmf_model1.fit_transform(X_link_origin)
    # nmf_model2=NMF(n_components=200)
    # X_text=nmf_model2.fit_transform(X_text_origin)

    # print X_link.shape, len(y_link),myu.sparse_matrix_norm_1_1(X_link)
    # print X_text.shape, len(y_text),myu.sparse_matrix_norm_1_1(X_text)

    # views_X = []
    # X1 = X_link.todense()
    # X2 = X_text.transpose().todense()
    # views_X.append(X1)
    # views_X.append(X2)
    # m, nn = X1.shape
    # print X1.shape
    # print X2.shape

    # views_weight = [0.7, 0.3]
    # views_U, views_V, V_consensus = multiNMF.multi_view_nmf(views_X, views_weight, nn, 100)
    # print V_consensus.shape
    #
    # X=preprocessing.normalize(V_consensus,norm="l1")
    # y=y_text
    #

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
    # clf_lg=LogisticRegression()
    # clf_lg.fit(X_train,y_train)
    # print clf_lg.score(X_test,y_test)

    # score_link=cross_validation.cross_val_score(clf_lg, X_link, y_link, cv=5,scoring='f1_macro')
    # print "link:",str(np.mean(score_link))
    #

    # score_text=cross_validation.cross_val_score(knn(), X_text, y_text, cv=5)
    # print "text:",str(np.mean(score_text))

    print "end", time.localtime()
