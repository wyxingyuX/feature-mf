# -*- coding:utf-8 -*-
import random
import numpy as np
import scipy.io as sio
from scipy.sparse import hstack
from scipy.sparse import vstack
from sklearn import cross_validation
from sklearn.decomposition import NMF
from sklearn.neighbors import KNeighborsClassifier as knn
import myutil as myu
from sklearn.utils import shuffle
from sklearn.linear_model import *
import MCL
import my_sim_nmf1
import my_sim_nmf2
import my_sim_nmf3
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split


def read_cate_id(src_cate_id_file, ignore_linenum=1):
    cate_id_dic = {}
    f = open(src_cate_id_file)
    line_num = 0
    for line in f:
        line_num += 1
        if line_num == ignore_linenum:
            continue
        elms = line.split()
        cate = elms[0].strip()
        id = elms[1].strip()
        cate_id_dic.setdefault(cate, []).append(id)
    f.close()
    return cate_id_dic


def gen_balance_data(src_cate_doc, link_file, per_cate_samples, dest_cate_doc):
    fl = open(link_file)
    have_link_id_dic = {}
    for line in fl:
        elms = line.split()
        id = elms[0].strip()
        have_link_id_dic[id] = id
    fl.close()

    src_cate_doc_dic = read_cate_id(src_cate_doc)
    wf = open(dest_cate_doc, "w")
    for k, v in src_cate_doc_dic.items():
        random.shuffle(v)
        sample_count = 0
        for id in v:
            if sample_count >= per_cate_samples:
                print sample_count
                break
            if have_link_id_dic.has_key(id):
                wf.write(k + "\t" + id + "\n")
                sample_count += 1

    wf.close()


def my_docid_code(src_file, dest_file):
    src_f = open(src_file)
    dest_f = open(dest_file, "w")
    line_num = 0
    docid_code_dic = {}
    for line in src_f:
        line_num += 1
        elms = line.split()
        id = elms[1]
        dest_f.write(id + "\t" + str(line_num) + "\n")
        docid_code_dic[id] = str(line_num)

    src_f.close()
    dest_f.close()
    return docid_code_dic


def cate_docidmycode(src_file, docid_mycode_dic, dest_file, ignore_linenum=1):
    src_f = open(src_file)
    dest_f = open(dest_file, "w")

    for line in src_f:
        elms = line.split()
        cate = elms[0]
        docid = elms[1]
        dest_f.write(cate + "\t" + docid_mycode_dic[docid] + "\n")
    src_f.close()
    dest_f.close()


def sel(src_file, keys, dest_file):
    fr = open(src_file)
    fw = open(dest_file, "w")
    for line in fr:
        elms = line.split()
        id = elms[0].strip()
        if keys.has_key(id):
            fw.write(line)
    fw.close()
    fr.close()


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
                word_code_dic[word] = str(len(word_code_dic) + 1)
    src_f.close()

    dest_f = open(dest_file, "w")
    for word, code in word_code_dic.items():
        dest_f.write(word + "\t" + str(code) + "\n")
    dest_f.close()
    return word_code_dic


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


def read_X_y(id_features_file, cate_id_file):
    docid_cate_dic = read_key_code_dic(cate_id_file, inverse=True)
    f = open(id_features_file)
    id_sparse_vec_dic = {}
    the_docid_cate_dic = {}
    for line in f:
        elms = line.split()
        docid = elms[0]
        vec = elms[1:]
        id_sparse_vec_dic[docid] = vec
        the_docid_cate_dic[docid] = docid_cate_dic[docid]
    f.close()
    X = myu.transf_csr_matrix(id_sparse_vec_dic, start_diem=1)

    y = range(len(the_docid_cate_dic))
    for k, v in the_docid_cate_dic.items():
        y[int(k) - 1] = v
    return X, y


def cate_count_havelink(cate_id_file, link_file):
    fl = open(link_file)
    have_link_id_dic = {}
    for line in fl:
        elms = line.split()
        id = elms[0].strip()
        have_link_id_dic[id] = id
    fl.close()

    cate_idhavelink_dic = {}
    fc = open(cate_id_file)
    line_num = 0
    for line in fc:
        line_num += 1
        if line_num == 1:
            continue
        elms = line.split()
        cate = elms[0].strip()
        id = elms[1].strip()
        if have_link_id_dic.has_key(id):
            cate_idhavelink_dic.setdefault(cate, []).append(id)
    fc.close()

    for k, v in cate_idhavelink_dic.items():
        print k, len(v)


def predata_dblp():
    base_path = "F:\ExpData\work"
    # cate_count_havelink(base_path+"\Cora\\answerkey7.list",base_path+"\Cora\\coralink.collection")
    gen_balance_data(base_path + "\DBLP\\answerkeys_indexed.list", base_path + "\DBLP\\dblplink.collection", 1000,
                     base_path + "\my\DBLP\\answerkeys_indexed.list")

    docid_mycode_dic = my_docid_code(base_path + "\my\DBLP\\answerkeys_indexed.list",
                                     base_path + "\my\DBLP\\docid_mycode.txt")

    sel(base_path + "\DBLP\\dblp.collection", docid_mycode_dic, base_path + "\my\DBLP\\dblp.collection")
    word_mycode_dic = word_mycode(base_path + "\my\DBLP\dblp.collection", base_path + "\my\DBLP\word_mycode.txt")
    text_sparse_vec(base_path + "\my\DBLP\\dblp.collection", docid_mycode_dic, word_mycode_dic,
                    base_path + "\my\DBLP\\text_sparse_vec.txt")

    sel(base_path + "\DBLP\\dblplink.collection", docid_mycode_dic, base_path + "\my\DBLP\\dblplink.collection")
    link_sparse_vec(base_path + "\my\DBLP\\dblplink.collection", docid_mycode_dic,
                    base_path + "\my\DBLP\\link_sparse_vec.txt")

    cate_docidmycode(base_path + "\my\DBLP\\answerkeys_indexed.list", docid_mycode_dic,
                     base_path + "\my\DBLP\\cate_docidmycode.txt")


def predata_cora():
    base_path = "F:\ExpData\work"
    gen_balance_data(base_path + "\Cora\\answerkey7.list", base_path + "\Cora\\coralink.collection", 200,
                     base_path + "\my\Cora\\answerkey7.list")


def dblp_item_feature_zn(src_file, docid_cate_dic, source_cate, dest_file):
    src_f = open(src_file)
    dest_f = open(dest_file, "w")
    for line in src_f:
        elms = line.split()
        doc_id = elms[0]
        if len(elms) <= 2:
            print doc_id
        if docid_cate_dic.has_key(doc_id):
            dest_f.write(doc_id)
            dest_f.write("\t\t" + docid_cate_dic[doc_id])
            dest_f.write("\t\t" + source_cate)
            total_words = int(elms[1])
            for i in range(0, 2 * total_words, 2):
                word = elms[i + 2]
                word_freq = int(elms[i + 3])
                for k in range(word_freq):
                    dest_f.write("\t" + word)
            dest_f.write("\n")
    src_f.close()
    dest_f.close()


def cora_link_feature_zn(src_file, docid_cate_dic, source_cate, dest_file):
    src_f = open(src_file)
    dest_f = open(dest_file, "w")
    for line in src_f:
        line = line.strip()
        elms = line.split()
        doc_id = elms[0]
        if docid_cate_dic.has_key(doc_id):
            dest_f.write(doc_id)
            dest_f.write("\t\t" + docid_cate_dic[doc_id])
            dest_f.write("\t\t" + source_cate)
            for i in range(1, len(elms), 2):
                link_id = elms[i].strip()
                link_w = elms[i + 1].strip()
                for k in range(int(link_w)):
                    dest_f.write("\t" + link_id)
            dest_f.write("\n")
    src_f.close()
    dest_f.close()


def cora_text_feature_zn(src_file, docid_cate_dic, source_cate, dest_file):
    src_f = open(src_file)
    dest_f = open(dest_file, "w")
    for line in src_f:
        elms = line.split()
        docid = elms[0]
        if docid_cate_dic.has_key(docid):
            dest_f.write(docid)
            dest_f.write("\t\t" + docid_cate_dic[docid])
            dest_f.write("\t\t" + source_cate)

            for w in elms[1:]:
                dest_f.write("\t" + w)
            dest_f.write("\n")
    src_f.close()
    dest_f.close()


def for_zn():
    # base_path = "F:\ExpData\work"
    # docid_cate_dic = read_key_code_dic(base_path + "\DBLP\\answerkeys_indexed.list", ignore_linenum=1, inverse=True)
    # dblp_item_feature_zn(base_path + "\DBLP\dblp.collection", docid_cate_dic, "text",
    #                      base_path + "\zn\DBLP\dblp.collection")
    # dblp_item_feature_zn(base_path + "\DBLP\dblplink.collection", docid_cate_dic, "link",
    #                      base_path + "\zn\DBLP\dblplink.collection")
    #
    # cora_docid_cate7_dic = read_key_code_dic(base_path + "\Cora\\answerkey7.list", ignore_linenum=1, inverse=True)
    # cora_link_feature_zn(base_path + "\Cora\\coralink.collection", cora_docid_cate7_dic, "link",
    #                      base_path + "\zn\Cora\coralink_7.collection")
    # cora_text_feature_zn(base_path + "\Cora\\cora.collection", cora_docid_cate7_dic, "text",
    #                      base_path + "\zn\Cora\cora_7.collection")
    #
    # cora_docid_cate18_dic = read_key_code_dic(base_path + "\Cora\\answerkey18.list", ignore_linenum=1, inverse=True)
    # cora_link_feature_zn(base_path + "\Cora\\coralink.collection", cora_docid_cate18_dic, "link",
    #                      base_path + "\zn\Cora\coralink_18.collection")
    # cora_text_feature_zn(base_path + "\Cora\\cora.collection", cora_docid_cate18_dic, "text",
    #                      base_path + "\zn\Cora\cora_18.collection")

    base_path="H:\webkb"
    docid_cate_dic = read_key_code_dic(base_path + "\\answerkey4.list_recode", ignore_linenum=1, inverse=True)
    cora_text_feature_zn(base_path + "\\webkbattrall.collection_recode", docid_cate_dic, "text",
                         base_path + "\\zn\\text.collection")
    dblp_item_feature_zn(base_path + "\\doclink.collection", docid_cate_dic, "link",
                         base_path + "\\zn\link.collection")


def gen_4_w2v(dblp_item_feature_file, dest_file, sort=False, ignore_center_word=False):
    fr = open(dblp_item_feature_file)
    contents_lines = []
    line_num = 0
    word_count = 0
    max_word = -1
    min_word = 10000
    for line in fr:
        line_num += 1
        contents = []
        elms = line.split()
        id = elms[0]
        if len(elms) <= 2:
            print id
        if not ignore_center_word:
            contents.append(id)
        total_words = int(elms[1])

        word_count += total_words
        if total_words < min_word:
            min_word = total_words
        if total_words > max_word:
            max_word = total_words

        words = []
        for i in range(0, 2 * total_words, 2):
            word = elms[i + 2]
            word_freq = int(elms[i + 3])
            for k in range(word_freq):
                words.append("\t" + word)
        if sort:
            words.sort()
        for w in words:
            contents.append(w)
        contents.append("\n")
        contents_lines.append(contents)
    fr.close()

    print "line_num:", line_num
    print "max_items:", max_word
    print "min_items:", min_word
    print "avg_items:", (1.0 * word_count) / line_num

    random.shuffle(contents_lines)

    fw = open(dest_file, "w")
    for contents in contents_lines:
        for w in contents:
            fw.write(w)
    fw.close()


def gen4wzv():
    base_path = "F:\ExpData\work"
    gen_4_w2v(base_path + "\my\DBLP\\dblp.collection", base_path + "\my\DBLP\\dblp_4_w2v.collection",
              ignore_center_word=True)
    gen_4_w2v(base_path + "\my\DBLP\\dblplink.collection", base_path + "\my\DBLP\\dblplink_4_w2v.collection", sort=True)


def check(X, y):
    base_path = "F:\ExpData\work"
    print "check start"
    gold_id_cate_dic = read_key_code_dic(base_path + "\my\DBLP\\cate_docidmycode.txt", inverse=True)
    for i in range(len(y)):
        gold = gold_id_cate_dic[str(i + 1)]
        if cmp(gold, y[i]):
            print i + 1, y[i], gold
    print "check end"


def read_X_y_3source(mtx_file, docmycode_cate_file):
    docmycode_cate_dic = read_key_code_dic(docmycode_cate_file)
    X = myu.matrix_market_2_csr_matrix(mtx_file)
    y = []
    colums = X.shape[1]
    for docmycode in range(1, colums + 1):
        y.append(int(docmycode_cate_dic[str(docmycode)]))
    return X, y


def read_doc_cate_retuers(file, ignore_line=0):
    fr = open(file)
    doc_cate_dic = {}
    line_num = 0
    for line in fr:
        line_num += 1
        if line_num <= ignore_line:
            continue
        doc_cate_dic[line_num] = int(line.strip())
    fr.close()
    return doc_cate_dic


def read_X_y_retuers(mtx_file, cate_file):
    doc_cate_dic = read_doc_cate_retuers(cate_file)
    X = myu.matrix_market_2_csr_matrix(mtx_file, ignore_line_num=2)
    y = []
    samples_num = X.shape[0]
    for doc in range(1, samples_num + 1):
        y.append(int(doc_cate_dic[doc]))
    return X, y


def bow_score(clf):
    base_path = "F:\ExpData\work"
    X_link_origin, y_link = read_X_y(base_path + "\my\DBLP\\link_sparse_vec.txt",
                                     base_path + "\my\DBLP\\cate_docidmycode.txt")
    print X_link_origin.shape, len(y_link)
    X_text_origin, y_text = read_X_y(base_path + "\my\DBLP\\text_sparse_vec.txt",
                                     base_path + "\my\DBLP\\cate_docidmycode.txt")
    print X_text_origin.shape, len(y_text)

    X_link = X_link_origin
    X_text = X_text_origin
    X_conc = hstack([X_text, X_link])
    print X_conc.shape

    # print "start save mat"
    # sio.savemat(base_path + "\my\DBLP\\dblp.mat", {'link': X_link_origin, "text": X_text_origin, "gnd": y_text})
    # print "end save mat"

    score_link = cross_validation.cross_val_score(clf, X_link, y_link, cv=5)
    print "bow link:", str(np.mean(score_link))

    score_text = cross_validation.cross_val_score(clf, X_text, y_text, cv=5)
    print "bow text:", str(np.mean(score_text))

    score_conc = cross_validation.cross_val_score(clf, X_conc, y_text, cv=5)
    print "bow conc:", str(np.mean(score_conc))


def nmf_score(clf):
    base_path = "F:\ExpData\work"

    X_link_origin, y_link = read_X_y(base_path + "\my\DBLP\\link_sparse_vec.txt",
                                     base_path + "\my\DBLP\\cate_docidmycode.txt")
    print X_link_origin.shape, len(y_link)

    X_text_origin, y_text = read_X_y(base_path + "\my\DBLP\\text_sparse_vec.txt",
                                     base_path + "\my\DBLP\\cate_docidmycode.txt")
    print X_text_origin.shape, len(y_text)

    nmf_model1 = NMF(n_components=20)
    X_link = nmf_model1.fit_transform(X_link_origin)
    print X_link.shape
    nmf_model2 = NMF(n_components=20)
    X_text = nmf_model2.fit_transform(X_text_origin)
    print X_text.shape
    X_conc = np.concatenate([X_text, X_link], axis=1)
    print X_conc.shape

    score_link = cross_validation.cross_val_score(clf, X_link, y_link, cv=5)
    print "nmf link:", str(np.mean(score_link))

    score_text = cross_validation.cross_val_score(clf, X_text, y_text, cv=5)
    print "nmf text:", str(np.mean(score_text))

    score_conc = cross_validation.cross_val_score(clf, X_conc, y_text, cv=5)
    print "nmf conc:", str(np.mean(score_conc))


def multi_nmf(clf):
    X_link_origin, y_link = read_X_y(base_path + "\my\DBLP\\link_sparse_vec.txt",
                                     base_path + "\my\DBLP\\cate_docidmycode.txt")
    print X_link_origin.shape, len(y_link)
    score = []
    for i in range(1, 4):
        V = sio.loadmat(base_path + "\my\DBLP\\V_centroid20_" + str(i) + ".mat")
        X = V["V_centroid_tmp"]

        score_tmp = cross_validation.cross_val_score(clf, X, y_link, cv=5)
        print i, "score:", np.mean(score_tmp)
        myu.a_append_bitems(score, score_tmp)
    print "multi_nmf", np.mean(score)


def clar(clf):
    base_path = "F:\ExpData\work"
    X, y = read_X_y(base_path + "\my\DBLP\\filled_link_sparse_vec.txt", base_path + "\my\DBLP\\cate_docidmycode.txt")
    print X.shape, len(y)

    score = cross_validation.cross_val_score(clf, X, y, cv=5)
    print "clar:", np.mean(score)


def mcl_score():
    base_path = "F:\ExpData\work"
    X_link_origin, y_link = read_X_y(base_path + "\my\DBLP\\link_sparse_vec.txt",
                                     base_path + "\my\DBLP\\cate_docidmycode.txt")
    print X_link_origin.shape, len(y_link)

    X_text_origin, y_text = read_X_y(base_path + "\my\DBLP\\text_sparse_vec.txt",
                                     base_path + "\my\DBLP\\cate_docidmycode.txt")
    print X_text_origin.shape, len(y_text)

    all_score_knn = []
    all_score_lr = []
    for ii in range(1, 6):
        print str(ii)
        # X_link_origin, y_link, X_text_origin, y_text = shuffle(X_link_origin, y_link, X_text_origin, y_text)

        X_link_origin_train, X_link_origin_test, y_link_train, y_link_test = cross_validation.train_test_split(
            X_link_origin, y_link, test_size=0.2, random_state=ii * 10)
        X_text_origin_train, X_text_origin_test, y_text_train, y_text_test = cross_validation.train_test_split(
            X_text_origin, y_text, test_size=0.2, random_state=ii * 10)

        X_link_origin = vstack([X_link_origin_train, X_link_origin_test])
        y_nolabel = [-1 for i in range(600)]
        y_link_label_nolabel = y_link_train + y_nolabel

        X_text_origin = vstack([X_text_origin_train, X_text_origin_test])
        y_text = y_text_train + y_text_test

        Nl = 2400
        nn = len(y_link)
        k = 20
        beta = 0.02
        r = 0.02

        views_X = []
        views_X.append(X_link_origin.transpose().todense())
        views_X.append(X_text_origin.transpose().todense())
        print "start mcl"
        view_U, V = MCL.mcl(views_X, y_link_label_nolabel, Nl, beta, r, nn, k)
        print "end mcl"

        print "start save mat"
        sio.savemat(base_path + "\my\DBLP\\mcl_V_" + str(ii) + ".mat", {'V': V})
        print "end save mat"

        V = V.transpose()
        V_train = V[0:2400, :]
        y_train = y_text[0:2400]

        V_test = V[2400:nn, :]
        y_test = y_text[2400:nn]

        clf = knn()
        clf.fit(V_train, y_train)
        score_knn = clf.score(V_test, y_test)
        all_score_knn.append(score_knn)
        print str(ii), "knn", score_knn

        clf_lr = LogisticRegression()
        clf_lr.fit(V_train, y_train)
        score_lr = clf_lr.score(V_test, y_test)
        all_score_lr.append(score_lr)
        print str(ii), "lr", score_lr

    print "mcl knn:", np.mean(all_score_knn)
    print "mcl lr:", np.mean(all_score_lr)


def my_sim_nmf_score():
    base_path = "F:\ExpData\work"

    # X_link_origin, y_link = read_X_y(base_path + "\my\DBLP\\link_sparse_vec.txt",
    #                                  base_path + "\my\DBLP\\cate_docidmycode.txt")
    # print X_link_origin.shape, len(y_link)
    #
    # X_text_origin, y_text = read_X_y(base_path + "\my\DBLP\\text_sparse_vec.txt",
    #                                  base_path + "\my\DBLP\\cate_docidmycode.txt")
    # print X_text_origin.shape, len(y_text)

    X_link_origin, y_link = read_X_y(base_path + "\my\DBLP\\doc_vec_link.txt",
                                     base_path + "\my\DBLP\\cate_docidmycode.txt")
    print X_link_origin.shape, len(y_link)
    X_text_origin, y_text = read_X_y(base_path + "\my\DBLP\\doc_vec_text.txt",
                                     base_path + "\my\DBLP\\cate_docidmycode.txt")
    print X_text_origin.shape, len(y_text)

    views_X = []
    views_X.append(X_link_origin.transpose().todense())
    views_X.append(X_text_origin.transpose().todense())

    X, views_U = my_sim_nmf1.sim_nmf(views_X, k=20)

    X = X.T
    y = y_link

    score_lr = cross_validation.cross_val_score(LogisticRegression(), X, y, cv=5)
    print "my sim_nmf lr:", str(np.mean(score_lr))
    score_knn = cross_validation.cross_val_score(knn(), X, y, cv=5)
    print "my sim_nmf knn:", str(np.mean(score_knn))


def w2v_score(clf):
    base_path = "F:\ExpData\work"
    X_link_origin, y_link = read_X_y(base_path + "\my\DBLP\\doc_vec_link.txt",
                                     base_path + "\my\DBLP\\cate_docidmycode.txt")
    print X_link_origin.shape, len(y_link)
    X_text_origin, y_text = read_X_y(base_path + "\my\DBLP\\doc_vec_text.txt",
                                     base_path + "\my\DBLP\\cate_docidmycode.txt")
    print X_text_origin.shape, len(y_text)

    X_link = X_link_origin
    X_text = X_text_origin
    X_conc = hstack([X_text, X_link])
    print X_conc.shape

    score_link = cross_validation.cross_val_score(clf, X_link, y_link, cv=5)
    print "w2v link:", str(np.mean(score_link))

    score_text = cross_validation.cross_val_score(clf, X_text, y_text, cv=5)
    print "w2v text:", str(np.mean(score_text))

    score_conc = cross_validation.cross_val_score(clf, X_conc, y_text, cv=5)
    print "w2v conc:", str(np.mean(score_conc))


def test_3source():
    base_path = "D:\ZMyDisk\BrowserDownload\\3sources"

    views_mtx = [base_path + "\\3sources\\3views_bbc.mtx", base_path + "\\3sources\\3views_guardian.mtx",
                 base_path + "\\3sources\\3views_reuters.mtx"]

    nmf_s = [[] for i in range(len(views_mtx))]
    my_sim_nmf_s = []
    K = 6
    for j in range(5):
        views_X = []
        y = []
        for i in range(len(views_mtx)):
            X, y = read_X_y_3source(views_mtx[i], base_path + "\\3sources\\3sources_mycode_cate.disjoint.clist")
            print X.shape, len(y)
            views_X.append(X.todense())

            X = X.T
            nmf_model1 = NMF(n_components=K)
            X = nmf_model1.fit_transform(X)

            kmeans = KMeans(n_clusters=K).fit(X)
            nmf_acc = myu.cluster_acc(kmeans.labels_, y)
            print str(j), str(i), " nmf k-means:", nmf_acc
            nmf_s[i].append(nmf_acc)
        X, views_U = my_sim_nmf1.sim_nmf(views_X, views_lamada=[0.1, 0.1, 0.1], k=K)

        # ohch，not ok
        # X, view_U, views_V = my_sim_nmf2.sim_nmf(views_X, views_lamada=[0.01, 0.01, 0.01],
        #                                          views_gamma=[0.001, 0.001, 0.001],k=K)
        # X, view_U, views_V = my_sim_nmf2.sim_nmf(views_X, views_lamada=[0, 0, 0],
        #                                          views_gamma=[0.1, 0.1, 0.1],alpha1=0,alpha2=0,beta=0,k=K,min_error=0.0001)

        X = X.T
        kmeans = KMeans(n_clusters=K).fit(X)
        my_sim_nmf_acc = myu.cluster_acc(kmeans.labels_, y)
        my_sim_nmf_s.append(my_sim_nmf_acc)
        print str(j), "mysim-nmf k-means:", my_sim_nmf_acc
    print "\n"
    for i in range(len(nmf_s)):
        print str(i), " nmf:", str(np.mean(nmf_s[i]))
    print "mysim_nmf:", str(np.mean(my_sim_nmf_s))


def test_reuters():
    base_path = "D:\ZMyDisk\BrowserDownload\ReutersEN\\reutersEN"
    views_mtx = [base_path + "\\reutersEN_5_EN.mtx", base_path + "\\reutersEN_5_FR.mtx",
                 base_path + "\\reutersEN_5_GR.mtx"]
    views_X = []
    y = None
    for i in range(len(views_mtx)):
        X, y = read_X_y_retuers(views_mtx[i], base_path + "\\reutersEN_act.txt")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=100)
        #
        X = X_train
        y = y_train
        # views_X.append(X)
        views_X.append(X.T.todense())
    # X0, X1, X2, y = shuffle(views_X[0], views_X[1], views_X[2], y)
    # views_X[0] = X0.T.todense()
    # views_X[1] = X1.T.todense()
    # views_X[2] = X2.T.todense()

    nmf_s = [[] for i in range(3)]
    my_sim_nmf_s = []
    K = 6
    for j in range(5):
        for i in range(len(views_X)):
            X = views_X[i]
            X = X.T
            print X.shape, len(y)

            nmf_model1 = NMF(n_components=K)
            X = nmf_model1.fit_transform(X)

            kmeans = KMeans(n_clusters=K).fit(X)
            nmf_acc = myu.cluster_acc(kmeans.labels_, y)
            print str(j), str(i), " nmf k-means:", nmf_acc
            nmf_s[i].append(nmf_acc)
        X, views_U = my_sim_nmf1.sim_nmf(views_X, views_lamada=[0.1, 0.1, 0.1], k=K)

        X = X.T
        kmeans = KMeans(n_clusters=K).fit(X)
        my_sim_nmf_acc = myu.cluster_acc(kmeans.labels_, y)
        my_sim_nmf_s.append(my_sim_nmf_acc)
        print str(j), "mysim-nmf k-means:", my_sim_nmf_acc
    print "\n"
    for i in range(len(nmf_s)):
        print str(i), " nmf:", str(np.mean(nmf_s[i]))
    print "mysim_nmf:", str(np.mean(my_sim_nmf_s))


def test_digit():
    handwritten = sio.loadmat("F:\Code\Matlab\Code_multiNMF\\handwritten.mat")
    views_X = []
    views_X.append(handwritten["fourier"].T)
    views_X.append(handwritten["pixel"].T)
    y_mat = handwritten["gnd"]
    y = [y_mat[i][0] for i in range(len(y_mat))]

    nmf_s = [[] for i in range(len(views_X))]
    my_sim_nmf_s = []
    K = 10
    for j in range(20):
        for i in range(len(views_X)):
            X = views_X[i]
            X = X.T
            print X.shape, len(y)

            nmf_model1 = NMF(n_components=K)
            X = nmf_model1.fit_transform(X)

            kmeans = KMeans(n_clusters=K).fit(X)
            nmf_acc = myu.cluster_acc(kmeans.labels_, y)
            print str(j), str(i), " nmf k-means:", nmf_acc
            nmf_s[i].append(nmf_acc)

        X, views_U = my_sim_nmf1.sim_nmf(views_X, views_lamada=[0.001, 0.001, 0.001], k=K)

        # oche, not ok
        # X, view_U, views_V = my_sim_nmf2.sim_nmf(views_X, views_lamada=[0.01, 0.01, 0.01],
        #                                          views_gamma=[0.1, 0.1, 0.1])

        X = X.T
        kmeans = KMeans(n_clusters=K).fit(X)
        my_sim_nmf_acc = myu.cluster_acc(kmeans.labels_, y)
        my_sim_nmf_s.append(my_sim_nmf_acc)
        print str(j), "mysim-nmf k-means:", my_sim_nmf_acc
    print "\n"
    for i in range(len(nmf_s)):
        print str(i), " nmf:", str(np.mean(nmf_s[i]))
    print "mysim_nmf:", str(np.mean(my_sim_nmf_s))


if __name__ == "__main__":
    print "hello"
    base_path = "F:\ExpData\work"
    clf_knn = knn()
    clf_lr = LogisticRegression()

    #predata_dblp()
    # gen4wzv()

    # bow_score(clf_lr)
    # nmf_score(clf_lr)
    # multi_nmf(clf_lr)

    # clar(clf_knn)
    # clar(clf_lr)
    # mcl_score()
    # my_sim_nmf_score()
    # w2v_score(knn())
    # test_3source()
    # test_reuters()
    # test_digit()
