import os,sys
import numpy as np
import pandas as pd

## save the stopword statistics for all corpora

## custom packages
src_dir = os.path.join(os.pardir,'src')
sys.path.append(src_dir)

from filter_words import run_stopword_statistics
from shi_json import shi_json_to_texts

# ## set up data
list_name_corpus = [
        'reuters_filter_10largest_dic',
        't20NewsGroup_topic_doc_no_short',
        'wos_topic_doc_delSci_withstop',
        'multi_lg_german',
        'multi_lg_portuguese_dw_general_v2',
        'multi_lg_chinese_sogou',
        ]

N_s = 1000

path_data = os.path.join(os.pardir,os.pardir,'data','s2019-05-29_stopword-corpora')

for corpus_name in list_name_corpus:
    print(corpus_name)
    ## define language for stopword list
    lang = 'en' ## default
    if 'german' in corpus_name:
        lang = 'de'
    if 'portuguese' in corpus_name:
        lang = 'pt'
    if 'chinese' in corpus_name:
        lang = 'cn' 

    path_stopword_list = os.path.join(path_data,'stopword_list_%s'%(lang))  ## path to stopword list


    fname_read = '%s.json'%(corpus_name)
    filename = os.path.join(path_data,fname_read)
    ## read hanyus json-format
    x_data = shi_json_to_texts(filename) ## contains the data
    ## vocabulary and lsit of texts
    list_words = x_data['list_w']
    list_texts = x_data['list_texts']
    dict_w_iw = x_data['dict_w_iw']
    list_labels = x_data['list_c']
    D = len(list_texts)

    ## get the statistics
    df = run_stopword_statistics(list_texts,N_s=N_s,path_stopword_list=path_stopword_list)

    ## save the statistics
    filename_save = os.path.join(os.pardir,'output','%s_stopword-statistics_Ns%s.csv'%(corpus_name,N_s))
    df.to_csv(filename_save)