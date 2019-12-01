import os, sys
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


def texts_nwd_csr(list_texts):
    '''
    Make a csr n_wd matrix from a list of texts.
    each text is a list of tokens.
    provide dict_w_iw == mapping of words to indices i_w=0,...,V-1
    '''

    ## unqiue words and alphabetically sorted list of words
    set_vocab = set([ token for doc in list_texts for token in doc ])
    list_w = sorted(list(set_vocab))
    V = len(list_w)
    dict_w_iw = dict(zip( list_w,np.arange(V)))


    D = len(list_texts)
    V = len(dict_w_iw)

    ## csr-format of the data
    rows = []
    cols = []
    data = []

    for i_doc, doc in enumerate(list_texts):
        data += [1]*len(doc)
        cols += [i_doc]*len(doc)
        rows += [ dict_w_iw[h] for h in doc]
    n_wd_csr = csr_matrix( ( data, (rows, cols) ), shape=(V, D), dtype=np.int64, copy=False)
    return n_wd_csr,dict_w_iw





