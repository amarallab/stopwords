import numpy as np
# import random
import os
import sys
# import gzip
from gensim import corpora
from collections import Counter
import subprocess
src_dir = os.path.abspath(os.path.join(os.pardir, 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# from common.convert_states import state_perturb_wd
from common.convert_states import nwd_to_texts, state_nwjd

def hdp_inference_wrapper(dict_input):
    '''
    Wrapper for hdp_inference

    Input:
        dict_input = {
            ## choose topic model
            'topic_model': 'hdp'

            ## provide corpus and number of topics if need
            , 'texts':texts

            ## optional, only works for synthetic corpus with token labeling
            , 'state_dwz_true': state_dwz
            , 'k_true': K

            ## optional
            , 'path_hdp': os.path.abspath(os.path.join(os.pardir,'src/external/hdp-bleilab/hdp-faster'))
        }

    Output:
        dict_output = {
              'p_td_infer': p_td ## p_td inferred by topic modle
            , 'token_labeling_nmi': nmi ## optional, results for token labeling, only for synthetic data
            , 'k_infer': number of topics inferred by topic model
        }
    '''
    # ############################
    # # Get input parameters
    texts = dict_input['texts']

    # # optional, only works for synthetic corpus with token labeling
    state_dwz_true = dict_input.get('state_dwz_true', None)
    k_true = dict_input.get('k_true', None)

    # # optional
    path_hdp = dict_input.get('path_hdp', os.path.abspath(os.path.join(os.pardir, 'src/external/hdp-bleilab/hdp-faster')))

    # # Call the true function:
    dict_output = hdp_inference_terminal(texts, state_dwz_true=state_dwz_true, k_true=k_true, path_hdp=path_hdp)

    return dict_output

def hdp_inference_terminal(texts, 
    state_dwz_true=None, 
    k_true=None, 
    path_hdp=os.path.abspath(os.path.join(os.pardir, 'src/external/hdp-bleilab/hdp-faster'))):
    '''
    Do the inference for p_dt and  state_dwz_ (optional)

    Input:

        ## provide corpus and number of topics if need
        , 'texts':texts

        ## optional, only works for synthetic corpus with token labeling
        , 'state_dwz_true': state_dwz
        , 'k_true': K

        ## optional
        , 'path_hdp': os.path.abspath(os.path.join(os.pardir,'src/external/hdp-bleilab/hdp-faster'))


    Output:
        dict_output = {
              'p_td_infer': p_td ## p_td inferred by topic modle
            , 'token_labeling_nmi': nmi ## optional, results for token labeling, only for synthetic data
            , 'k_infer': number of topics inferred by topic model
        }
    '''

    #############################
    # # Generate a empty dic for output
    dict_output = {}

    # ############################
    # # inference for p_dt

    train_dir = make_path_tmp_hdp()
    train_fname = texts_corpus_hdp(texts, train_dir)
    dir_cwd = os.getcwd()
    os.chdir(path_hdp)
    cmd_hdp = './hdp --train_data %s --directory %s' % (train_fname, train_dir)
    p = subprocess.Popen(cmd_hdp, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()
    os.chdir(dir_cwd)
    # # doc-topic counts
    f = open(train_dir + 'final.doc.states', 'r')
    x = f.readlines()
    f.close()
    D_ = len(x)
    K_hdp = len(x[0].split())
    p_td_hdp = np.zeros((D_, K_hdp))
    for i_d, d in enumerate(x):
        n_j_tmp = np.array([int(h_) for h_ in d.split()])
        p_td_hdp[i_d, :] = n_j_tmp / float(np.sum(n_j_tmp))
    # os.system('rm -rf %s'%( train_dir))

    dict_output['p_td_infer'] = p_td_hdp

    # ############################
    # # get the number of topics:
    f = open(train_dir + 'final.topics', 'r')
    x = f.readlines()
    f.close()
    k_hdp = len(x)
    dict_output['k_infer'] = k_hdp

    # ############################
    # # individual labels
    f = open(train_dir + 'final.word-assignments', 'r')
    header = f.readline()
    x = f.readlines()
    f.close()
    state_dwz_hdp = [tuple([int(h_) for h_ in h.split()]) for h in x]
    dict_output['state_dwz_infer'] = state_dwz_hdp

    # ##############
    # # infer p_wt

    all_word_list = [i[1] for i in state_dwz_hdp]
    n_w = max(all_word_list) + 1

    num_k = k_hdp
    p_wt_infer = np.zeros([n_w, num_k])
    for i in state_dwz_hdp:
        tmp_w = i[1]
        tmp_t = i[2]
        p_wt_infer[tmp_w, tmp_t] += 1
    p_wt_infer = p_wt_infer / p_wt_infer.sum(axis=0)
    dict_output['p_wt_infer'] = p_wt_infer

    os.system('rm -rf %s' % (train_dir))

    return dict_output


def make_path_tmp_hdp():
    marker = True
    while marker is True:
        ind_tmp = np.random.randint(0, 32000)
        path_tmp = os.path.abspath(os.path.join(os.pardir, 'tmp'))
        path_tmp_name = os.path.join(path_tmp, 'tmp_hdp_%s' % (str(ind_tmp)))
        if os.path.isdir(path_tmp_name):
            pass
        else:
            os.system('cd %s;mkdir tmp_hdp_%s' % (str(path_tmp), str(ind_tmp)))
            marker = False
    return path_tmp_name + '/'


def texts_corpus_hdp(texts, train_dir):
    fname_data = 'data_tmp'
    f = open(train_dir + fname_data, 'w')
    for text in texts:
        c_text = Counter(text)
        V_d = len(c_text)
        list_w_text = list(c_text.keys())
        list_w_text.sort()
        f.write(str(V_d))
        for w_ in list_w_text:
            f.write(' ' + str(w_) + ':' + str(c_text[w_]))
        f.write('\n')
    f.close()
    return train_dir + fname_data
