# import models.modelfront


def topicmodel_inference_front(dict_input):
    chosen_model = dict_input['topic_model']

    if chosen_model == 'sbm':
        # graph-tool's stochastic block model
        from models.gt import sbm_inference_wrapper
        dict_output = sbm_inference_wrapper(dict_input)

    elif chosen_model == 'ldags':
        # mallet's lda
        from models.ldags import ldags_inference_wrapper
        dict_output = ldags_inference_wrapper(dict_input)

    elif chosen_model == 'ldavb':
        # gensim's lda
        from models.ldavb import ldavb_inference_wrapper
        dict_output = ldavb_inference_wrapper(dict_input)

    elif chosen_model == 'hdp':
        # hierarchical dirichlet process
        from models.hdp import hdp_inference_wrapper
        dict_output = hdp_inference_wrapper(dict_input)

    elif chosen_model == 'tm':
        # topic mapping
        from models.tm import tm_inference_wrapper
        dict_output = tm_inference_wrapper(dict_input)

    return dict_output


def formulate_input_for_topicModel(
        dict_output_corpus, topic_model,
        K_pp=None, V_pp=None,
        input_k=None, flag_lda_opt=0, flag_coherence=0):
    '''
    Formulate the input dictionary for each topic model algorithm:
    ldavb, ldags, hdp, tm, sbm

    Input:
    dict_output_corpus =
    {
    'texts_list_shuffle':       ## the corpus
    'state_dwz_shuffle':        ## list of tuple,
                do not exist for real-world corpus
    }
    topic_model: str, the name of topic model
    K_pp: int, real/true number of topics
    V_pp: int, number of word types in the vocabulary
    input_k:    int, input number of topics for models
                (e.g., ldavb, ldags) which needs is as a parameter,
                if not given, set input_k = K_pp
    flag_lda_opt: b

    Output:
    dict_input_topicmodel=
    {
    'topic_model':   ## the name of topic model
    'texts':
    ....
    }   ## the output is model-specific
    '''

    if input_k is None:
        input_k = K_pp

    texts_list_shuffle = dict_output_corpus['texts']
    state_dwz_shuffle = dict_output_corpus.get('state_dwz', None)

    dict_input_topicmodel = {}
    dict_input_topicmodel['topic_model'] = topic_model
    dict_input_topicmodel['flag_coherence'] = flag_coherence

    if topic_model == 'ldavb':
        dict_input_topicmodel['texts'] = texts_list_shuffle
        dict_input_topicmodel['input_k'] = input_k

        if state_dwz_shuffle is not None:
            dict_input_topicmodel['state_dwz_true'] = state_dwz_shuffle
            dict_input_topicmodel['k_true'] = K_pp
            dict_input_topicmodel['input_v'] = V_pp
        if flag_lda_opt == 1:
            dict_input_topicmodel['set_alpha'] = 'auto'
            dict_input_topicmodel['set_beta'] = 'auto'

    if topic_model == 'ldags':
        dict_input_topicmodel['texts'] = texts_list_shuffle
        dict_input_topicmodel['input_k'] = input_k

        if state_dwz_shuffle is not None:
            dict_input_topicmodel['state_dwz_true'] = state_dwz_shuffle
            dict_input_topicmodel['k_true'] = K_pp
        if flag_lda_opt == 1:
            dict_input_topicmodel['dN_opt'] = 10

    if topic_model == 'hdp':
        dict_input_topicmodel['texts'] = texts_list_shuffle

        if state_dwz_shuffle is not None:
            dict_input_topicmodel['state_dwz_true'] = state_dwz_shuffle
            dict_input_topicmodel['k_true'] = K_pp

    if topic_model == 'tm':
        dict_input_topicmodel['texts'] = texts_list_shuffle

        if state_dwz_shuffle is not None:
            dict_input_topicmodel['state_dwz_true'] = state_dwz_shuffle
            dict_input_topicmodel['k_true'] = K_pp

    if topic_model == 'sbm':
        dict_input_topicmodel['texts'] = texts_list_shuffle

        if state_dwz_shuffle is not None:
            dict_input_topicmodel['state_dwz_true'] = state_dwz_shuffle
            dict_input_topicmodel['k_true'] = K_pp

    return dict_input_topicmodel
