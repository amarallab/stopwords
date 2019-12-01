import numpy as np
import copy
from gensim import corpora
import json



def tranfer_real_corpus_toID_and_shuffle(real_corpus_rStop_rShort_list, all_doc_class_rStop_rShort_array):
    texts_list = real_corpus_rStop_rShort_list
    topic_list = all_doc_class_rStop_rShort_array

    ##########################################################
    # Change the tokens into ids
    ##########################################################

    # save the corpora with words
    texts_list_word = copy.deepcopy(texts_list)

    # Change the words into ids
    texts_list_tmp = copy.deepcopy(texts_list_word)
    dict_gs = corpora.Dictionary(texts_list_tmp)
    new_dict_token2id = dict(dict_gs.token2id)

    for document_id in range(len(texts_list_tmp)):
        for token_id in range(len(texts_list_tmp[document_id])):
            token_tmp = texts_list_tmp[document_id][token_id]
            texts_list_tmp[document_id][token_id] = str(new_dict_token2id[token_tmp])

    # Save all the ids back to oright list
    texts_list = copy.deepcopy(texts_list_tmp)

    # Shuffle texts_list and topic_list
    topic_arr = np.array(topic_list)
    texts_arr = np.array(texts_list)
    arr = np.arange(len(topic_list))
    np.random.shuffle(arr)
    shuffle_topic_list = list(topic_arr[arr])
    shuffle_texts_list = list(texts_arr[arr])

    return shuffle_texts_list, shuffle_topic_list
