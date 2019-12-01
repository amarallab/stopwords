import numpy as np
from collections import Counter
# import sys
# import os
# import string
import random


def state_nwjd(state_dwz_, D_, V_, K_):
    '''convert a dwz-state to count matrices:
    In: - state_dwz_, list  [(doc-id,word-id,topic-id)]
            with len= number of tokens in corpus
        - D_, number of documents
        - V_, number of word-types
        - K_, number of topics
    Out:
        - n_wd_, word-document counts
        - n_wj_, word-topic counts
        - n_jd_, topic-document counts
    '''
    n_wd_ = np.zeros((V_, D_))
    n_wj_ = np.zeros((V_, K_))
    n_jd_ = np.zeros((K_, D_))
    c_dwz_ = Counter(state_dwz_)
    for dwz_, n_dwz_ in c_dwz_.items():
        d_ = dwz_[0]
        w_ = dwz_[1]
        z_ = dwz_[2]
        n_wd_[w_, d_] += n_dwz_
        n_wj_[w_, z_] += n_dwz_
        n_jd_[z_, d_] += n_dwz_
    return n_wd_, n_wj_, n_jd_


def state_perturb(state_dwz_, p_):
    '''Shuffle a fraction p_ the topic labels of a dwz-state
    In: - state_dwz_, list  [(doc-id,word-id,topic-id)]
            with len= number of tokens in corpus
        - p_, fraction of topic-labels to shuffle
    Out:
        - shuffled dwz-state
    '''
    n_perturb_ = int(len(state_dwz_) * p_)  # number of labels to randomize
    ind_perturb_ = np.arange(len(state_dwz_))
    random.shuffle(ind_perturb_)
    state_dwz_new_ = list(state_dwz_)

    z_select_ = [state_dwz_[i_][2] for i_ in ind_perturb_[:n_perturb_]]
    np.random.shuffle(z_select_)

    for i_, i_ind_ in enumerate(ind_perturb_[:n_perturb_]):
        dwz_ = state_dwz_[i_ind_]
        z_new_ = z_select_[i_]
        dwz_new_ = (dwz_[0], dwz_[1], z_new_)
        state_dwz_new_[i_ind_] = dwz_new_
    return state_dwz_new_


def state_perturb_wd(state_dwz_):
    '''Shuffle all the topic labels for the same (doc-id,word-id)-tokens
    In:
    - state_dwz_, list [(doc-id,word-id,topic-id)]
                with len= number of tokens in corpus
    Out:
        - shuffled dwz-state
    '''
    state_dwz_sorted_ = sorted(
        state_dwz_, key=lambda tup: (tup[0], tup[1], tup[2]))
    state_dwz_sorted_r_ = []
    tup_dwz_tmp_ = (-1, -1, -1)
    state_dwz_tmp_ = []
    for tup_dwz_ in state_dwz_sorted_:
        if tup_dwz_[0] != tup_dwz_tmp_[0] or tup_dwz_[1] != tup_dwz_tmp_[1]:
            random.shuffle(state_dwz_tmp_)
            state_dwz_sorted_r_ += state_dwz_tmp_
            state_dwz_tmp_ = []
            tup_dwz_tmp_ = tup_dwz_
        state_dwz_tmp_ += [tup_dwz_]
    random.shuffle(state_dwz_tmp_)
    state_dwz_sorted_r_ += state_dwz_tmp_
    return state_dwz_sorted_, state_dwz_sorted_r_


def nwd_to_texts(n_wd):
    '''convert a n_wd-count matrix to actual texts
    In: - n_wd, number of tokens of word w in doc d
    Out:
        - texts, a list of len=# of docs, i.e. texts = [text1, text2, ... textD] with
          texti is a list of len=textlength, i.e. texti=[w1, w2, ..]
    '''
    texts = []
    D = len(n_wd[0, :])
    for d in range(D):
        text = []
        for i_w, n_w in enumerate(n_wd[:, d]):
            if n_w > 0:
                text += [str(i_w)] * int(n_w)
        texts += [text]
    return texts


def texts_to_nwd(texts, V):
    '''convert a list of list ([ doc1, doc2,..., docD ] with
        doci = [w1, w10, w17] as a list of tokens  )
    In: - texts, list of list
        - V, int; number of word-types ( the tokens appearing can be one of 0,1,...,V-1 )
    Out:
        - n_wd, number of tokens of word w in doc d; shape = VxD
    '''
    D = len(texts)
    n_wd_rand = np.zeros((V, D)).astype('int')
    for d in range(D):
        d = int(d)
        c_tmp = Counter(texts[d])
        for w, n in c_tmp.items():
            w = int(w)
            n_wd_rand[w, d] = n
    return n_wd_rand


def order_block(list_t_w_, p_wj_, K1, K2):
    list_ind_j = []
    for j in np.arange(K1):
        inds_ = np.where(np.array(list_t_w_) == j)[0]
        x_ = np.sum(p_wj_[inds_, :], axis=0)
        ind_j_ = np.argmax(x_)
        if ind_j_ not in list_ind_j:
            list_ind_j += [ind_j_]
    list_ind_j_full = list(range(K2))
    list_ind_j_diff = list(set(list_ind_j_full).difference(set(list_ind_j)))
    list_ind_j += list_ind_j_diff
    return list_ind_j
