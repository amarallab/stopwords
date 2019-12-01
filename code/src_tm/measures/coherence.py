# from measures.coherence import

import numpy as np
from common.convert_states import state_nwjd

def topic_coherence_from_state_dwz(state_dwz,input_k,n=10, eps=1.0):
    D = len(set([dwz[0] for dwz in state_dwz])) ## number of documents
    V = len(set([dwz[1] for dwz in state_dwz])) 
    n_wd_, n_wj_, n_jd_ = state_nwjd(state_dwz, D, V, input_k)
    C = topic_cherence_C(n_wd_, n_wj_, n=n,eps=eps)
    return np.mean(C)

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


def topic_cherence_C_tf(n_wd, n_wj, n=10, eps=1.0):
    '''Calculates an extension of Mimnos topic coherence for topics inferred from (any)
       topic model. Instead of document-frequencies we calculate number of word-cooccurrences in a topic and
       compre with a null model based on random shuffling.
    In: - n_wd, the corpus represented as a count matrix number of times word w appears in document d
        - n_wj, the inferred word-topic matrix, counts how often word-type w is labelled as topic j
          --> used to get the top M words
        - n, number of words used in evaluation of each topic (the top-n words of each topic)
        - epsilon, smoothin parameter (default=1). some say it is better to use 10**(-12)
        pass
    Out:
        - coherence score for each topic, array len=number of topics
    '''
    # find topn words for each topic
    list_topn = topic_find_topnw(n_wj, n)
    K_ = len(list_topn)

    list_C_M_t = []
    # eps = 10.0**(-12) # ddowney uses 10^{-12}

    # N = np.sum(n_wd)
    n_d = np.sum(n_wd, axis=0)

    p_w_ = np.sum(n_wd, axis=1) / float(np.sum(n_wd))
    for t in range(K_):
        list_w = list_topn[t]
        C_M_t = 0.0
        # C_M_t_rand = 0.0
        tmp_n = 0
        for m_ in range(2, n + 1, 1):
            for l_ in range(1, m_, 1):
                tmp_n += 1
                i_m = m_ - 1
                i_l = l_ - 1
                w_m = list_w[i_m]
                w_l = list_w[i_l]
                df_m_l = np.sum(n_wd[w_m, :] * n_wd[w_l, :])
                df_m_l_rand = np.sum(n_d * (n_d - 1.0)) * p_w_[w_m] * p_w_[w_l]
                C_M_t += np.log(((df_m_l + eps)) / (df_m_l_rand))
                # if df_m_l > 0:
                #     C_M_t += df_m_l / np.sum(n_d * (n_d - 1.0)) * np.log((df_m_l) / (df_m_l_rand))

        list_C_M_t += [C_M_t]
    return np.array(list_C_M_t) / tmp_n
