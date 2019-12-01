
# import this pacakge: from models.ldags import

# system packages
import gensim as gs
from gensim import corpora
import numpy as np
import os
import gzip

# private packages
from models.ldamallet import *
# from measures.overlap import get_dict_output_token_labeling
from common.convert_states import state_nwjd
# from measures.perplexity import get_train_test_corpus
# from measures.coherence import topic_cherence_C


def ldags_inference_wrapper(dict_input):
    '''
    Wrapper for ldags_inference

    Input:
        dict_input = {
            ## choose topic model
            'topic_model': 'ldags'

            ## provide corpus and number of topics if need
            , 'texts':texts
            , 'input_k': K

            ## optional, only works for synthetic corpus with token labeling
            , 'state_dwz_true': state_dwz
            , 'k_true': K

            ## optional
            , 'input_v': V  # only need for 'ldavb' token labeling
            , 'path_mallet': os.path.abspath(os.path.join(os.pardir,'src/external/mallet-2.0.8RC3/bin/mallet'))
            , 'dN_opt':0
            , 'iterations':1000
        }

    Output:
        dict_output = {
            'p_td_infer': p_td ## p_td inferred by topic modle
            , 'token_labeling_nmi': nmi ## optional, results for token labeling, only for synthetic data
        }
    '''

    # # Get input parameters
    texts = dict_input['texts']
    input_k = dict_input['input_k']

        # # optional, only works for synthetic corpus with token labeling
    state_dwz_true = dict_input.get('state_dwz_true', None)
    k_true = dict_input.get('k_true', None)

    # # optional
    input_v = dict_input.get('input_v', None)
    path_mallet = dict_input.get(
        'path_mallet',
        os.path.abspath(os.path.join(os.pardir, 'src/external/mallet-2.0.8RC3/bin/mallet')))

    # dN_opt will be assigned to optimize_interval in the ldags-mallet algorithm: This option turns on hyperparameter optimization, which allows the model to better fit the data by allowing some topics to be more prominent than others. Optimization every 10 iterations is reasonable.
    dN_opt = dict_input.get('dN_opt', 0)

    # # optional： iteration limit & hyper-parameter
    iterations = dict_input.get('iterations', 1000)
    alpha = dict_input.get('set_alpha', 50.0)
    beta = dict_input.get('set_beta', 0.01)

    # # Call the true function:

    dict_output = ldags_inference_terminal(
        texts, input_k, state_dwz_true=state_dwz_true,
        k_true=k_true, input_v=input_v,
        path_mallet=path_mallet,
        dN_opt=dN_opt, iterations=iterations,
        alpha=alpha, beta=beta)

    return dict_output



def ldags_inference_terminal(
        texts, input_k, state_dwz_true=None,
        k_true=None, input_v=None,
        path_mallet=None,
        dN_opt=0, iterations=1000,
        alpha=50.0, beta=0.01):
    '''
    Do the inference for p_dt and  state_dwz_ (optional)

    Input:
        ## provide corpus and number of topics if need
        , 'texts':texts
        , 'input_k': K

        ## optional, only works for synthetic corpus with token labeling
        , 'state_dwz_true': state_dwz
        , 'k_true': K

        ## optional
        , 'input_v': V  # only need for 'ldavb' token labeling
        , 'path_mallet': os.path.abspath(os.path.join(os.pardir,'src/external/mallet-2.0.8RC3/bin/mallet'))
        , 'dN_opt':0
        , 'iterations':1000

    Output:
        dict_output = {
            'p_td_infer': p_td ## p_td inferred by topic modle
            , 'token_labeling_nmi': nmi ## optional, results for token labeling, only for synthetic data
        }
    '''

    # # Generate a empty dic for output
    dict_output = {}

    # # inference for p_dt
    if input_v is not None:
        # # for the synthetic corpurs with token labeling
        dict_gs = gs.corpora.Dictionary([[str(i)] for i in range(input_v)])
    else:
        # # for real-world corpus and the synthetic corpurs without token labeling
        dict_gs = corpora.Dictionary(texts)

    corpus_gs = [dict_gs.doc2bow(text) for text in texts]

    D = len(texts)
    path_tmp = make_path_tmp_lda()
    model = LdaMallet(
        path_mallet, corpus_gs,
        num_topics=input_k,
        id2word=dict_gs,
        prefix=path_tmp,
        iterations=iterations,
        optimize_interval=dN_opt,
        workers=1,
        alpha=alpha,
        beta=beta)

    # print("iterations limit: %s" % (iterations))

    # <<< infer p(t|d)
    fdoctopics_path = model.fdoctopics()
    with open(fdoctopics_path, "r") as text_file:
        lines = text_file.readlines()
    p_d_t_ldamallet = np.zeros([D, input_k])

    for d_num in range(D):
        t_d_oneline_str = lines[d_num]
        t_d_oneline_list = t_d_oneline_str.strip('\n').split('\t')[2:]
        for t_num in range(input_k):
            p_d_t_ldamallet[d_num, t_num] = t_d_oneline_list[t_num]

    dict_output['p_td_infer'] = p_d_t_ldamallet
    # >>>

    # <<< Get the nmi for token_labeling
    fname_labels = path_tmp + 'state.mallet.gz'
    state_dwz_infer, alpha_, beta_ = state_read_mallet(fname_labels)
    # print('set_LDAGS_alpha: %s, set_LDAVB_eta: %s' % (alpha_[0], beta_))
    # print('length_LDAGS_alpha: %s, length_LDAVB_eta: %s' % (len(alpha_), 1))

    # if state_dwz_true is not None:
    #     # nmi = state_dwz_nmi(state_dwz_true, state_dwz_infer, k_true, input_k)

    #     dict_output_token_labeling = get_dict_output_token_labeling(state_dwz_true, state_dwz_infer, k_true, input_k)
    #     dict_output.update(dict_output_token_labeling)

    # In general, we do not need to output state_dwz_infer
    dict_output['state_dwz_infer'] = state_dwz_infer
    # >>>

    # <<< infer p(w|t)
    D = max(np.array(state_dwz_infer)[:, 0]) + 1
    V = max(np.array(state_dwz_infer)[:, 1]) + 1
    K = input_k
    n_wd_infer, n_wj_infer, n_jd_infer = state_nwjd(state_dwz_infer, D, V, K)

    beta_tmp = beta
    beta_array = np.ones([V, 1]) * beta_tmp
    n_wj_beta_array = n_wj_infer + beta_array
    n_wj_beta_array_vector = np.sum(n_wj_infer + beta_array, axis=0)

    p_wt_infer = n_wj_beta_array / n_wj_beta_array_vector
    dict_output['p_wt_infer'] = p_wt_infer
    # >>>

    os.system('rm -rf %s' % (path_tmp))

    return dict_output


def make_path_tmp_lda():
    '''Make a temporary folder for all the data that is going back and forth to mallet.
    Out:
        - path to the temporary folder
    '''
    marker = True
    while marker is True:
        ind_tmp = np.random.randint(0, 32000)
        path_tmp = os.path.abspath(os.path.join(os.pardir, 'tmp'))
        path_tmp_name = os.path.join(path_tmp, 'tmp_lda_%s' % (str(ind_tmp)))
        if os.path.isdir(path_tmp_name):
            pass
        else:
            os.system('cd %s;mkdir tmp_lda_%s' % (str(path_tmp), str(ind_tmp)))
            marker = False
    return path_tmp_name + '/'


# INTERFACE TO THE MALLET-VERSION OF LDA
def state_read_mallet(file):
    '''Read the inferred labeled dwz-state (and the possibly optimized hyperparameters)
       from the mallet-inference located in the temporary folder
    In: - filename with ending 'state.mallet.gz' in temporary folder defined by make_path_tmp_lda()
    Out:
        - state_tmp, inferred dwz-state list, the tokens are in the same order as state_dwz
        - alpha_auto, doc-hyperparameter vector (if hyperparameter optimization was on this is the last value), array len=K
        - beta_auto, word-hyperparameter, float
    '''
    f = gzip.open(file, 'r')
    x_header = f.readline()
    x_alpha = f.readline()
    x_beta = f.readline()
    x_data = f.readlines()
    f.close()
    alpha_auto = np.array(x_alpha.split()[2:]).astype('float')
    beta_auto = float(x_beta.split()[2])
    state_tmp = [(int(h.split()[0]), int(h.split()[4]), int(h.split()[5])) for h in x_data]

    return state_tmp, alpha_auto, beta_auto
