import os, sys
import numpy as np
import json

def shi_json_to_texts(f_json):
    with open(f_json,'r') as f:
        x = json.load(f)

    ## categories
    list_c = list(x.keys())

    ## the vocabulary and text-selction
    D = 0
    V = 0
    N = 0
    set_words = set([])
    list_texts = []
    list_labels = []
    for i_c,c in enumerate(list_c):
        list_texts_c = x[c]
        for doc in list_texts_c:
            set_words.update(set(doc))
            D += 1
            N += len(doc)
            list_texts += [doc]
            list_labels+=[c]
            
    list_w = sorted(list(set_words))
    V = len(list_w)

    list_id = np.arange(D)
    list_iw = np.arange(V)
    dict_vocab_w_iw = dict(zip( list_w,list_iw ))

    result = {}
    result['list_texts'] = list_texts
    result['dict_w_iw'] = dict_vocab_w_iw
    result['list_w'] = list_w
    result['list_c'] = list_labels
    return result





