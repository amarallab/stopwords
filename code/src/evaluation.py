import numpy as np
from collections import Counter
from random import shuffle
import random



def topic_cherence_C(n_wd, n_wj, n=10, eps=1.0):
    '''
    Calculates Mimnos topic coherence for topics inferred from (any) topic model.
    In:
        - n_wd, the corpus represented as a count matrix number of times word w appears in document d
        - n_wj, the inferred word-topic matrix, counts how often word-type w is labelled as topic j
          --> used to get the top M words
        - n, number of words used in evaluation of each topic (the top-n words of each topic)
        - epsilon, smoothin parameter (default=1). some say it is better to use 10**(-12)
        pass
    Out:
        - coherence score for each topic, array len=number of topics
    '''

    max_x = n_wd.shape[0]
    if max_x < n:
        n = max_x

    # find topn words for each topic
    list_topn = topic_find_topnw(n_wj, n)
    K_ = len(list_topn)

    list_C_M_t = []
    # eps = 10.0**(-12) # ddowney uses 10^{-12}

    # p_w_ = np.sum(n_wd, axis=1) / float(np.sum(n_wd))
    for t in range(K_):
        list_w = list_topn[t]

        C_M_t = 0.0
        tmp_n = 0
        for m_ in range(2, n + 1, 1):
            # print(m_)
            for l_ in range(1, m_, 1):
                tmp_n += 1
                # print(m_,l_)
                i_m = m_ - 1
                i_l = l_ - 1
                w_m = list_w[i_m]
                w_l = list_w[i_l]
                df_m_l = np.sum((n_wd[w_m, :] > 0) * (n_wd[w_l, :] > 0))
                df_l = np.sum((n_wd[w_l, :] > 0))
                # df_m = np.sum((n_wd[w_m, :] > 0))
                C_M_t += np.log((df_m_l + eps) / df_l)

        list_C_M_t += [C_M_t]

    return np.array(list_C_M_t) / tmp_n


def topic_find_topnw(n_wj_, n_):
    # # find topn words for each topic
    list_topn = []
    K_ = len(n_wj_[0, :])
    for t in range(K_):
        # # the simple argsort will always give the same order in case of a tie
        # ind_sort = list(np.argsort(n_wj_[:,t])[::-1])

        # instead use lexsort to give random indices in case of a tie
        a = n_wj_[:, t]
        b = np.random.random(a.size)
        ind_sort = np.lexsort((b, a))[::-1]

        list_topn += [ind_sort[:n_]]
    return list_topn


def obtain_nmi_unsup(topic_list, p_td_ldavb_array, removed_topic_list=None):
    '''
    Obtain nmi with unsupervised methods
    Input:
    - topic_list: list, list of topics for all documents
    - p_td_ldavb_array: array, topic proportion for each documents
    Output:
    - nmi: normalized mutual information

    '''
    list_t_d_pred = predict_topic_p_td_unsup(p_td_ldavb_array)

    if removed_topic_list is not None and len(removed_topic_list) > 0:
        num_doc_removed = len(removed_topic_list)
        k_num = max(topic_list)
        random_topic_removed = list(np.random.choice(k_num, num_doc_removed))

        topic_list = list(topic_list) + list(removed_topic_list)
        list_t_d_pred = list(list_t_d_pred) + random_topic_removed

    nmi = calc_class_doc_nmi(topic_list, list_t_d_pred)

    return nmi


def predict_topic_p_td_unsup(p_dt):
    '''predict topic of document based on maximum in topic-doc distribution
    IN: p_d_t: doc-topic distribution D x K
    OUT: list of predicted topics len(D) with entries in {0,1,...,K-1} 
    '''
    list_doc_topic = []
    D = len(p_dt[:, 0])
    for i_d in range(D):
        t = np.argmax(p_dt[i_d])
        list_doc_topic += [t]
    return list_doc_topic


def calc_class_doc_nmi(list_t_true, list_t_pred):
    '''
    Claculate norm mutual information between true and predicted topics of documents
    IN: two list of len=D where each entry is the true and pred topic of the respective document
    OUT: nmi (float)
    '''
    K1 = max(list_t_true) + 1  # number of topics in true
    K2 = max(list_t_pred) + 1  # number of topics in pred
    N = len(list_t_true)
    n_tt = np.zeros((K1, K2))
    list_z1_z2 = [(list_t_true[i], list_t_pred[i]) for i in range(N)]
    c_z1_z2 = Counter(list_z1_z2)
    for z1_z2_, n_z1_z2_ in c_z1_z2.items():
        n_tt[z1_z2_[0], z1_z2_[1]] += n_z1_z2_
    p_tt = n_tt / float(N)
    p_t1 = np.sum(p_tt, axis=1)
    p_t2 = np.sum(p_tt, axis=0)
    H1 = sum([-p_ * np.log(p_) for p_ in p_t1 if p_ > 0.0])
    H2 = sum([-p_ * np.log(p_) for p_ in p_t2 if p_ > 0.0])
    MI = 0.0
    for i_ in range(K1):
        for j_ in range(K2):
            p1_ = p_t1[i_]
            p2_ = p_t2[j_]
            p12_ = p_tt[i_, j_]
            if p12_ > 0.0:
                MI += p12_ * np.log(p12_ / (p1_ * p2_))
    NMI = 2.0 * MI / (H1 + H2)
    return NMI



def state_dwz_nmi(state_dwz1_, state_dwz2_, K1_, K2_, normalized=True):
    '''
    Calculate the normalized mutual information (NMI) between two labeled dwz-states, i.e. how much the two labeled states overlap.
    In: - state_dwz1_,
        - state_dwz2_,
        - K1_, # of topics in state_dwz1_, int
        - K2_, # of topics in state_dwz2_, int
    Out:
        - NMI, float
    '''
    # VI_ = 0.0
    N_ = len(state_dwz1_)
    n_tt_ = np.zeros((K1_, K2_))

    state_dwz1_s, state_dwz1_ = state_perturb_wd(state_dwz1_)  # sort and shuffle labels across same words and docs
    state_dwz2_s, state_dwz2_ = state_perturb_wd(state_dwz2_)  # sort and shuffle labels across same words and docs

    list_z1_z2_ = [(state_dwz1_[i_][2], state_dwz2_[i_][2]) for i_ in range(N_)]
    c_z1_z2_ = Counter(list_z1_z2_)
    for z1_z2_, n_z1_z2_ in c_z1_z2_.items():
        n_tt_[z1_z2_[0], z1_z2_[1]] += n_z1_z2_
    p_tt_ = n_tt_ / float(N_)
    p_t1_ = np.sum(p_tt_, axis=1)
    p_t2_ = np.sum(p_tt_, axis=0)
    H1_ = sum([-p_ * np.log(p_) for p_ in p_t1_ if p_ > 0.0])
    H2_ = sum([-p_ * np.log(p_) for p_ in p_t2_ if p_ > 0.0])
    MI_ = 0.0
    for i_ in range(K1_):
        for j_ in range(K2_):
            p1_ = p_t1_[i_]
            p2_ = p_t2_[j_]
            p12_ = p_tt_[i_, j_]
            if p12_ > 0.0:
                MI_ += p12_ * np.log(p12_ / (p1_ * p2_))
    if normalized is True:
        NMI_ = 2.0 * MI_ / (H1_ + H2_)
    else:
        NMI_ = 1.0 * MI_
        # if we want to return the unnormalized mutual information
    return NMI_

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