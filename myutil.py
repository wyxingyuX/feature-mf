# -*- coding:utf-8 -*-
from scipy.sparse import csr_matrix


def matrix_norm_1_1(mat):
    m, n = mat.shape
    print mat.shape
    result = 0
    for i in range(m):
        for j in range(n):
            # print mat.get(i,j)
            result += abs(mat[i][j])
    return result


def sparse_matrix_norm_1_1(csr):
    norm = 0
    next = 0
    j = 0
    for i in range(len(csr.indptr) - 1):
        next = csr.indptr[i + 1]
        start = j
        while j < (start + next - csr.indptr[i]):
            # row \t col (have value)
            row = i + 1
            col = csr.indices[j] + 1
            norm += abs(csr.data[col])
            j += 1
    return norm


def transf_csr_matrix_vecs(w2vs):
    data = []
    rows = []
    cols = []
    line_count = 0
    for i in range(len(w2vs)):
        w2v = w2vs[i]
        for diem in w2v:
            elms = diem.split(":")
            rows.append(line_count)
            cols.append(int(elms[0]))
            data.append(float(elms[1]))
        line_count += 1
    return csr_matrix((data, (rows, cols)))


def transf_csr_matrix(id_w2v_dic, start_diem=1):
    data = []
    rows = []
    cols = []
    for i, w2v in id_w2v_dic.items():
        for diem in w2v:
            elms = diem.split(":")
            rows.append(int(i) - 1)
            cols.append(int(elms[0]) - start_diem)
            data.append(float(elms[1]))
    # print rows, cols
    return csr_matrix((data, (rows, cols)))


def a_append_bitems(a, b):
    for item in b:
        a.append(item)
    return a


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


def matrix_market_2_csr_matrix(file, ignore_line_num=0):
    data = []
    rows = []
    cols = []

    fr = open(file)
    line_num = 0
    for line in fr:
        line_num += 1
        if line_num <= ignore_line_num:
            continue
        elms = line.split()
        row = elms[0].strip()
        col = elms[1].strip()
        d = elms[2].strip()

        rows.append(int(row) - 1)
        cols.append(int(col) - 1)
        data.append(float(d))
    fr.close()
    return csr_matrix((data, (rows, cols)))


def cluster_label_2_cate(cluster_labels, y_labels):
    cluster_label_cate_ids_dic = {}
    for i in range(len(cluster_labels)):
        cluster_label = cluster_labels[i]
        y_label = y_labels[i]
        cluster_label_cate_ids_dic.setdefault(cluster_label, dict()).setdefault(y_label, []).append(i)

    cluster_2_cate_dic = {}
    for cluster, cate_ids_dic in cluster_label_cate_ids_dic.items():
        cate_ids_arry = sorted(cate_ids_dic.items(), key=lambda d: len(d[1]), reverse=True)
        cate = cate_ids_arry[0][0]
        cluster_2_cate_dic[cluster] = cate
    return cluster_2_cate_dic


def cluster_acc(cluster_labels, y_labels):
    cluster_label_2_cate_dic = cluster_label_2_cate(cluster_labels, y_labels)
    correct_count = 0
    for i in range(len(cluster_labels)):
        cluster_label = cluster_labels[i]
        p_y = cluster_label_2_cate_dic[cluster_label]
        y_label = y_labels[i]
        if cmp(p_y, y_label) == 0:
            correct_count += 1
    acc = 1.0 * correct_count / len(cluster_labels)
    return acc

# def get_parent_path(abs_file):
#     elms=abs_file.split("\\")
#     if len(elms)==0:
#         return abs_file
#     par_path=""
#     for i in range(len(elms)-1):
#         par_path+=elms[i]
#         par_path+="\\\\"
#     return par_path
