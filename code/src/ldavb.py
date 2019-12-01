# system package
import gensim as gs
from gensim import corpora
import numpy as np
import random
from scipy.special import psi
import timeit
from memory_profiler import memory_usage

from evaluation import topic_cherence_C


def ldavb_inference_terminal(
        texts, input_k, state_dwz_true=None,
        k_true=None, input_v=None,
        dN_opt=0, minimum_probability=0,
        iterations=50, gamma_threshold=0.001,
        alpha='symmetric', eta=None, flag_coherence=0):
    '''
    Do the inference for p_dt and  state_dwz_ (optional)

    Input:

        ## provide corpus and number of topics if need
        'texts':texts
        'input_k': K

        ## optional, only works for synthetic corpus with token labeling
        'state_dwz_true': state_dwz
        'k_true': K
        'input_v': V  # only need for ldavb token labeling

        ## optional
        'dN_opt':0 ## optional
        'minimum_probability':0 ## optional

    Output:
        dict_output = {
              'p_td_infer': p_td ## p_td inferred by topic modle
            , 'token_labeling_nmi': nmi ## optional, results for token labeling, only for synthetic data
        }
    '''

    # Generate a empty dic for output
    dict_output = {}

    # inference for p_dt
    if input_v is not None:
        dict_gs = gs.corpora.Dictionary([[str(i)] for i in range(input_v)])
    else:
        dict_gs = corpora.Dictionary(texts)

    corpus_gs = [dict_gs.doc2bow(text) for text in texts]

    lda_g = gs.models.ldamodel.LdaModel(
        corpus_gs, id2word=dict_gs,
        num_topics=input_k,
        alpha=alpha, eta=eta,
        eval_every=dN_opt, iterations=iterations,
        gamma_threshold=gamma_threshold,
        minimum_probability=minimum_probability)

    # Get the topic distribution for each document from the ldavb
    # and shuffle the generating process
    D = len(texts)
    p_td_ldavb_array = np.zeros([D, input_k])
    a_tem = list(range(len(corpus_gs)))
    random.shuffle(a_tem)

    for i_d in a_tem:
        one_doc_corpus = corpus_gs[i_d]
        p_oned_t = lda_g.get_document_topics(one_doc_corpus)
        for a, b in p_oned_t:
            p_td_ldavb_array[i_d, a] = b

    dict_output['p_td_infer'] = p_td_ldavb_array

    # Get the p_wt_infer
    all_terms = list(dict_gs.iterkeys())
    V = len(all_terms)
    p_wt_infer = np.zeros([V, input_k])

    for tmp_t in range(input_k):
        p_w_tmpT = lda_g.show_topic(tmp_t, topn=V)
        for tuple_p_tmpW_tmpT in p_w_tmpT:
            tmp_w = int(tuple_p_tmpW_tmpT[0])
            tmp_p = (tuple_p_tmpW_tmpT[1])
            p_wt_infer[tmp_w, tmp_t] = tmp_p
    dict_output['p_wt_infer'] = p_wt_infer

    # get coherence
    if flag_coherence == 1:
        n_wd = np.zeros((V, D))
        for tmp_doc_id, tmp_doc in enumerate(texts):
            for tmp_word in tmp_doc:
                n_wd[int(tmp_word), tmp_doc_id] += 1

        dict_output['coherence'] = topic_cherence_C(n_wd, p_wt_infer)

    # get state_dwz_infer
    if state_dwz_true is None:
        lambda_knu = lda_g.state.get_lambda()  # (KxV)
        gamma_dk = lda_g.inference(corpus_gs)[0]  # (D,K)

        state_dwz_true_fake = []
        for tmp_doc_id, tmp_doc in enumerate(corpus_gs):
            for tmp_token_num in tmp_doc:
                tmp_token = tmp_token_num[0]
                tmp_occurrence = tmp_token_num[1]
                state_dwz_true_fake += [(tmp_doc_id, tmp_token, 0)] * tmp_occurrence

        state_dwz_ldavb_infer, n_wj_ldavb, n_jd_ldavb = state_dwz_LDAVB(state_dwz_true_fake, lambda_knu, gamma_dk)
        dict_output['state_dwz_infer'] = state_dwz_ldavb_infer

    # Get the nmi for token_labeling
    if state_dwz_true is not None:
        lambda_knu = lda_g.state.get_lambda()  # (KxV)
        gamma_dk = lda_g.inference(corpus_gs)[0]  # (D,K)

        state_dwz_ldavb_infer, n_wj_ldavb, n_jd_ldavb = state_dwz_LDAVB(
            state_dwz_true, lambda_knu, gamma_dk)

        dict_output_token_labeling = get_dict_output_token_labeling(
            state_dwz_true, state_dwz_ldavb_infer, k_true, input_k)

        dict_output['state_dwz_infer'] = state_dwz_ldavb_infer

        dict_output.update(dict_output_token_labeling)

    return dict_output


#############################################
# INTERFACE TO THE GENSIM-VERSION OF LDA
#############################################
def state_dwz_LDAVB(state_dwz_, lambda_knu_, gamma_dk_):
    '''Infer the individual token-labels for the inferred topic-doc and word-topic matrices from gensim.
    In: - state_dwz_, list  [(doc-id,word-id,topic-id)]  with len= number of tokens in corpus
        - lambda_knu_, inferred word-topic matrix from gensim, array (KxV)
        - gamma_dk_, inferred doc-topic matrix from gensim, array (D,K)
    Out:
        - state_dwz_ldavb, inferred dwz-state list, the tokens are in the same order as state_dwz
    '''
    lambda_k = np.sum(lambda_knu_, axis=1)  # (K,)
    psi_lambda_k = psi(lambda_k)
    psi_gamma_dk = psi(gamma_dk_)

    K_ = len(lambda_knu_[:, 0])
    V_ = len(lambda_knu_[0, :])
    D_ = len(gamma_dk_[:, 0])
    n_wj_ = np.zeros((V_, K_)).astype('int')
    n_jd_ = np.zeros((K_, D_)).astype('int')

    state_dwz_infer = []
    for dwz in state_dwz_:
        d = dwz[0]
        w = dwz[1]
        p_z = np.exp(psi_gamma_dk[d, :] + psi(lambda_knu_[:, w]) - psi_lambda_k)
        p_z /= np.sum(p_z)
        z_infer = np.argmax(np.random.multinomial(1, p_z))
        state_dwz_infer += [(d, w, z_infer)]
        n_wj_[w, z_infer] += 1
        n_jd_[z_infer, d] += 1
    return state_dwz_infer, n_wj_, n_jd_


def obtain_ldavb_cpuTime_memory(texts, input_k,):

    start_time = timeit.default_timer()
    mem_usage = memory_usage((ldavb_inference_cpuTime_memory, (texts, input_k, )))
    elapsed_time = timeit.default_timer() - start_time

    increment_memory = float(max(mem_usage)) - float(min(mem_usage))

    return elapsed_time, increment_memory


def ldavb_inference_cpuTime_memory(
        texts, input_k,
        k_true=None, input_v=None,
        dN_opt=0, minimum_probability=0,
        iterations=50, gamma_threshold=0.001,
        alpha='symmetric', eta=None):
    '''
    Do the inference for p_dt and  state_dwz_ (optional)

    Input:

        ## provide corpus and number of topics if need
        'texts':texts
        'input_k': K

        ## optional, only works for synthetic corpus with token labeling
        'state_dwz_true': state_dwz
        'k_true': K
        'input_v': V  # only need for ldavb token labeling

        ## optional
        'dN_opt':0 ## optional
        'minimum_probability':0 ## optional

    Output:
        dict_output = {
              'p_td_infer': p_td ## p_td inferred by topic modle
            , 'token_labeling_nmi': nmi ## optional, results for token labeling, only for synthetic data
        }
    '''

    # Generate a empty dic for output

    # inference for p_dt
    if input_v is not None:
        dict_gs = gs.corpora.Dictionary([[str(i)] for i in range(input_v)])
    else:
        dict_gs = corpora.Dictionary(texts)

    corpus_gs = [dict_gs.doc2bow(text) for text in texts]

    lda_g = gs.models.ldamodel.LdaModel(
        corpus_gs, id2word=dict_gs,
        num_topics=input_k,
        alpha=alpha, eta=eta,
        eval_every=dN_opt, iterations=iterations,
        gamma_threshold=gamma_threshold,
        minimum_probability=minimum_probability)

    # mem_usage = memory_usage((gs.models.ldamodel.LdaModel, (corpus_gs, id2word=dict_gs, num_topics=input_k, alpha=alpha, eta=eta, eval_every=dN_opt, iterations=iterations, gamma_threshold=gamma_threshold, minimum_probability=minimum_probability)))

    # increment_memory = float(max(mem_usage)) - float(min(mem_usage))

    return 1
