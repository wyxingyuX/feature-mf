# -*- coding:utf-8 -*-
import numpy as np
import scipy.io as sio
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import *

import myutil as myu


def my_docid_code(src_file, dest_file):
    src_f = open(src_file)
    dest_f = open(dest_file, "w")
    line_num = 0
    docid_code_dic = {}
    for line in src_f:
        if line_num == 0:
            line_num += 1
            continue
        elms = line.split()
        id = elms[1]
        dest_f.write(id + "\t" + str(line_num) + "\n")
        docid_code_dic[id] = str(line_num)
        line_num += 1

    src_f.close()
    dest_f.close()
    return docid_code_dic


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
        dest_f.write(docid_mycode_dic[docid] + "\t" + cate + "\n")
    src_f.close()
    dest_f.close()


def link_sparse_vec(src_file, docid_mycode_dic, dest_file):
    src_f = open(src_file)
    dest_f = open(dest_file, "w")
    for line in src_f:
        line = line.strip()
        elms = line.split()
        doc_id = elms[0]
        if docid_mycode_dic.has_key(doc_id):
            dest_f.write(docid_mycode_dic[doc_id])
            for i in range(1, len(elms), 2):
                link_id = elms[i].strip()
                link_w = elms[i + 1].strip()
                if docid_mycode_dic.has_key(link_id):
                    dest_f.write("\t" + docid_mycode_dic[link_id] + ":" + link_w)
            dest_f.write("\n")
    src_f.close()
    dest_f.close()


def read_key_code_dic(src_file):
    src_f = open(src_file)
    key_code_dic = {}
    for line in src_f:
        elms = line.split()
        key = elms[0].strip()
        code = elms[1].strip()
        if not key_code_dic.has_key(key):
            key_code_dic[key] = code
    src_f.close()
    return key_code_dic


def read_text_X_y(src_file, docid_mycode_dic, docidmycode_cate_dic):
    f = open(src_file)
    X_corpus = []
    y = []
    mycodes = []
    for line in f:
        elms = line.split(None, 1)
        docid = elms[0]
        words = elms[1]
        if docid_mycode_dic.has_key(docid):
            docid_mycode = docid_mycode_dic[docid]
            cate = docidmycode_cate_dic[docid_mycode]
            mycodes.append(docid_mycode)
            y.append(cate)
            X_corpus.append(words)
    f.close()
    vecer = CountVectorizer()
    X = vecer.fit_transform(X_corpus)
    return X, y, mycodes

def read_X_y(id_features_file, id_cate_file):
    docid_cate_dic = read_key_code_dic(id_cate_file)
    f = open(id_features_file)
    sparse_vecs = []
    y = []
    for line in f:
        elms = line.split()
        docid = elms[0]
        vec = elms[1:-1]
        sparse_vecs.append(vec)
        y.append(docid_cate_dic[docid])
    f.close()
    X = myu.transf_csr_matrix(sparse_vecs)
    return X, y

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


if __name__ == "__main__":
    print "hello"
    base_path = "F:\ExpData\work\\"
    # docid_mycode_dic=my_docid_code(base_path+"Cora\\answerkey7.list",base_path+"my\\Cora\\7\\docid_mycode.txt")
    # link_sparse_vec(base_path+"Cora\\coralink.collection",docid_mycode_dic,base_path+"my\\Cora\\7\\link_sparse_vec.txt")
    docid_mycode_dic = read_key_code_dic(base_path + "my\\Cora\\7\\docid_mycode.txt")
    docidmycode_cate(base_path + "Cora\\answerkey7.list", docid_mycode_dic,
                     base_path + "my\\Cora\\7\\docidmycode_cate.txt")
    docidmycode_cate_dic = read_key_code_dic(base_path + "my\\Cora\\7\\docidmycode_cate.txt")

    cate_count(base_path + "my\Cora\\7\\docidmycode_cate_index.txt")
    X_link, y_link = read_X_y(base_path + "my\Cora\\7\\link_sparse_vec.txt",
                              base_path + "my\Cora\\7\\docidmycode_cate_index.txt")
    X_text, y_text, mycodes_text = read_text_X_y(base_path + "Cora\\cora.collection", docid_mycode_dic,
                                                 docidmycode_cate_dic)
    # print "start save mat"
    # sio.savemat(base_path+"my\Cora\\7\\cora.mat",{'link':X_link,"text":X_text})
    # print "end save mat"

    # print "link:",X_link.shape, len(y_link),myu.sparse_matrix_norm_1_1(X_link)
    # print "text:",X_text.shape, len(y_text),myu.sparse_matrix_norm_1_1(X_text)
    # views_X = []
    # # X1 = X_link.todense()
    # X2 = X_text.transpose().todense()
    # # views_X.append(X1)
    # views_X.append(X2)
    # # m, nn = X1.shape
    # # print X1.shape
    # m,nn=X2.shape
    # print X2.shape
    #
    # views_weight = [1]
    # views_U, views_V, V_consensus = multiNMF.multi_view_nmf(views_X, views_weight, nn, 100)
    # print V_consensus.shape
    #
    # X = preprocessing.normalize(V_consensus, norm="l1")
    # y = y_text
    #

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
    clf_lg = LogisticRegression()

    # score_link = cross_validation.cross_val_score(clf_lg, X_link, y_link, cv=5, scoring='f1_macro')
    # print "link:", str(np.mean(score_link))

    score_text = cross_validation.cross_val_score(clf_lg, X_text, y_text, cv=5, scoring='f1_macro')
    print "text:", str(np.mean(score_text))

    # clf_lg.fit(X_train, y_train)
    # print clf_lg.score(X_test, y_test)
