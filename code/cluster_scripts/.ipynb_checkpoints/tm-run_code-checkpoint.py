import os,sys
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import time
## run topic model on corpus with variable filtering of stopwords

###################
## custom packages
###################

src_dir = os.path.join(os.pardir,'src')
sys.path.append(src_dir)

from filter_words import make_stopwordlist
from filter_words import remove_stopwords_from_list_texts
from shi_json import shi_json_to_texts

## tools from the topic-modeling
path_tm = os.path.join(os.pardir,'src_tm') 
sys.path.append(path_tm)
from models.modelfront import topicmodel_inference_front
from models.modelfront import formulate_input_for_topicModel
from corpora.realcorpora import tranfer_real_corpus_toID_and_shuffle
from measures.overlap import state_dwz_nmi
from measures.coherence import topic_coherence_from_state_dwz

from sklearn.metrics import v_measure_score
from sklearn.metrics import mutual_info_score

###################
## parameters
###################

## parameters of data
# corpus
corpus_name = sys.argv[1]# 'reuters_filter_10largest_dic'
# number of random realization from random null model
## parameters from stopwords
N_s = int(sys.argv[2]) #1000
method = sys.argv[3] #'INFOR'
# cutoff_val = 0.9 ## we iterate over an array

## parameters from topic model
topic_model = sys.argv[4] #'hdp' ## which algorithm to use

n_rep = int(sys.argv[5]) #2 # 10

path_save = sys.argv[6]

k_set = int(sys.argv[7]) ## 20
print(sys.argv)

## parameters set manually
cutoff_type =  'p'
arr_p = np.linspace(0.0,0.9,10)
# arr_p = [0.9]
path_data = os.path.join(os.pardir,os.pardir,'data','s2019-05-29_stopword-corpora')
path_output = os.path.join(os.pardir,'output')

###################
## Run model
###################

## stopword statistics
filename = os.path.join(path_output,'%s_stopword-statistics_Ns%s.csv'%(corpus_name,N_s))
df_stop = pd.read_csv(filename,index_col=0,na_filter = False)
df_stop['manual']=df_stop['manual'].replace(to_replace='',value=np.nan).replace(to_replace='1.0',value=1.0)

## data
fname_read = '%s.json'%(corpus_name)
filename = os.path.join(path_data,fname_read)
## read hanyus json-format
x_data = shi_json_to_texts(filename) ## contains the data
## vocabulary and lsit of texts
list_words = x_data['list_w']
list_texts = x_data['list_texts']
list_labels = x_data['list_c']
D = len(list_texts)
V = len(list_words)

for i_p,p in enumerate(arr_p):
    cutoff_val = p


    ## make stopwords
    list_stopwords = make_stopwordlist(df_stop,
                                      method = method,
                                      cutoff_type = cutoff_type, 
                                      cutoff_val = cutoff_val, )
    ## remove all words from stopword list
    list_texts_filter = remove_stopwords_from_list_texts(list_texts, list_stopwords)
    N = sum([ len(doc) for doc in list_texts ])
    N_filter = sum([ len(doc) for doc in list_texts_filter ])
    V_filter = len(set([ token  for doc in list_texts_filter for token in doc]))

    ## remove articles that end up with 0 tokens
    n_d =  np.array([len(h) for h in list_texts_filter])
    ind_nd_0 = np.where(n_d==0)[0]
    ind_nd_n0 = np.where(n_d>0)[0]

    list_texts_filter_n0 = [list_texts_filter[i] for i in ind_nd_n0]
    list_labels_n0 = [list_labels[i] for i in ind_nd_n0]

    ## convert words to ids
    shuffle_texts_list, shuffle_topic_list = tranfer_real_corpus_toID_and_shuffle(
        list_texts_filter_n0, 
        list_labels_n0)

    ## setup the topic model
    dict_output_corpus = {
        'texts': shuffle_texts_list,
    }
    K_pp = len(set(list_labels)) ## number of different labels in document metadata.
    dict_input_topicModel = formulate_input_for_topicModel(
        dict_output_corpus, 
        topic_model, 
        K_pp=K_pp, 
        input_k=k_set, 
        flag_lda_opt=1, flag_coherence=1,
        )
    # only for mallet
    if topic_model == 'ldags':
        dict_input_topicModel['path_mallet'] = os.path.abspath(os.path.join(path_tm,'external','mallet-2.0.8RC3','bin','mallet'))
        dict_input_topicModel['dN_opt'] = 10
    elif topic_model == 'hdp':
        dict_input_topicModel['path_hdp'] = os.path.abspath(os.path.join(path_tm,'external','hdp-bleilab','hdp-faster'))


    ## we run the topic model n_rep + 1 times  in order to calculate reproducibility
    dict_output_topicModel2 = topicmodel_inference_front(dict_input_topicModel)
    state_dwz_infer2 = dict_output_topicModel2['state_dwz_infer']
    k_infer2 = dict_output_topicModel2.get('k_infer',k_set)

    dict_result = defaultdict(list)
    dict_result['p'] = cutoff_val ## planned fraction of tokens to removed
    dict_result['p-emp'] = 1.-N_filter/N ## actual fraction of tokens removed
    for i_n_rep in range(n_rep):
        print(p, i_n_rep)
        ## run topic model
        t1 = time.time()
        dict_output_topicModel = topicmodel_inference_front(dict_input_topicModel)
        t2 = time.time()
        dict_result['t'] += [t2-t1] ## record how long it takes for the topic model inference

        k_infer = dict_output_topicModel.get('k_infer',k_set) ## get number of inferred topics (hdp) otherwise preselceted number of topics
        p_td_infer = dict_output_topicModel['p_td_infer']
        state_dwz_infer = dict_output_topicModel['state_dwz_infer']

        ## accuracy
        # true and predicted labels from the articles with length >0
        list_labels_true = list(shuffle_topic_list)
        list_labels_pred = list(np.argmax(p_td_infer,axis=1))
        # append articles with length 0, the predicted label is drawn randomly
        for i_d in ind_nd_0:
            list_labels_true += [list_labels[i_d]]
            list_labels_pred += [np.random.choice(k_infer)]
        nmi_acc = v_measure_score(list_labels_true,list_labels_pred)
        dict_result['nmi_acc'] += [nmi_acc]

        ## reproducibility
        nmi_repr = state_dwz_nmi(state_dwz_infer,state_dwz_infer2,k_infer,k_infer2)
        # current state becomes reference state
        state_dwz_infer2 = state_dwz_infer.copy()
        k_infer2 = int(k_infer)
        dict_result['nmi_repr'] += [nmi_repr]

        ## mutual information between topic-docs and topics-words
        list_d = [dwz[0] for dwz in state_dwz_infer]
        list_t = [dwz[2] for dwz in state_dwz_infer]
        list_w = [dwz[1] for dwz in state_dwz_infer]
        # nmi and mi for topics-docs
        mi_td = mutual_info_score(list_t,list_d)
        nmi_td = v_measure_score(list_t,list_d)
        # nmi and mi for topics-words
        mi_tw = mutual_info_score(list_t,list_w)
        nmi_tw = v_measure_score(list_t,list_w)
        dict_result['mi_td'] += [mi_td]
        dict_result['nmi_td'] += [nmi_td]
        dict_result['mi_tw'] += [mi_tw]
        dict_result['nmi_tw'] += [nmi_tw]
        
        ## coherence
        coh = topic_coherence_from_state_dwz(state_dwz_infer,k_infer)
        dict_result['coh'] += [coh]

    # print(dict_result)
    fname_save = '%s_%s-%s_%s_p%.2f_Ns%s_nrep%s'%(corpus_name,topic_model,k_set,method,cutoff_val,N_s,n_rep)
    fname_save +='.json'
    filename = os.path.join(path_save,fname_save)
    with open(filename, 'w') as outfile:
        json.dump(dict_result, outfile)