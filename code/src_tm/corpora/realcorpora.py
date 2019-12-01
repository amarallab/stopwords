# from corpora.realcorpora import get_topic_text_reuter_wordid_withstop


# <<< Add the src path to sys.path
import sys
import os

containing_folder = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(containing_folder, os.pardir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# >>>


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


def get_raw_real_world_corpus(which_data):

    ##########################################################
    # Get data from reuters
    ##########################################################

    containing_folder = os.path.dirname(os.path.abspath(__file__))
    data_save_path = os.path.abspath(os.path.join(containing_folder, os.pardir, os.pardir, 'data', 'reuters', 'data_filter'))

    if which_data == 0:
        data_save_file = 'reuters_filter_10more_dic.json'
        which_file = 'reuters_10more'
    if which_data == 1:
        data_save_file = 'reuters_filter_10largest_dic.json'
        which_file = 'reuters_10largest'
    if which_data == 2:
        data_save_file = 'reuters_filter_top_odd10_dic.json'
        which_file = 'reuters_top_odd10'
    if which_data == 3:
        data_save_file = 'reuters_filter_top_even10_dic.json'
        which_file = 'reuters_top_even10'
    if which_data == 4:
        data_save_file = 'rcv1_doc_oneTopic.json'
        which_file = 'rcv1_1topic'
    if which_data == 5:
        data_save_file = 'rcv1_doc_oneTopic_v2.json'
        which_file = 'rcv1_1topic_v2'

    if which_data == 6:
        data_save_file = 't20NewsGroup_topic_doc_all_terms.json'
        which_file = 't20NewsGroup_topic_doc_all_terms'
    if which_data == 7:
        data_save_file = 't20NewsGroup_topic_doc_no_short.json'
        which_file = 't20NewsGroup_topic_doc_no_short'
    if which_data == 8:
        data_save_file = 't20NewsGroup_topic_doc_no_stop.json'
        which_file = 't20NewsGroup_topic_doc_no_stop'
    if which_data == 9:
        data_save_file = 't20NewsGroup_topic_doc_stemmed.json'
        which_file = 't20NewsGroup_topic_doc_stemmed'

    if which_data == 10:
        data_save_file = 'rcv1_category_2_labels_10pct.json'
        which_file = 'rcv1_category_2_labels_10pct'
    if which_data == 11:
        data_save_file = 'rcv1_category_2_nonover_labels_20pct.json'
        which_file = 'rcv1_category_2_nonover_labels_20pct'
    if which_data == 12:
        data_save_file = 'wos_topic_doc.json'
        which_file = 'wos_topic_doc'
    if which_data == 13:
        data_save_file = 'wos_topic_doc20length.json'
        which_file = 'wos_topic_doc20length'
    if which_data == 14:
        data_save_file = 'wos_topic_doc5length.json'
        which_file = 'wos_topic_doc5length'

    if which_data == 15:
        data_save_file = 'wos_topic_doc20length_random.json'
        which_file = 'wos_topic_doc20length_random'
    if which_data == 16:
        data_save_file = 'wos_topic_doc5length_random.json'
        which_file = 'wos_topic_doc5length_random'

    if which_data == 17:
        data_save_file = 'wos_topic_90doc1topic_length80_small_test.json'
        which_file = 'wos_topic_small_test_600doc'

    if which_data == 18:
        data_save_file = 'wos_topic_doc_delSci.json'
        which_file = 'wos_topic_doc_delSci'
    if which_data == 19:
        data_save_file = 'wos_topic_doc20length_random_delSci.json'
        which_file = 'wos_topic_doc20length_random_delSci'

    if which_data == 20:
        data_save_file = 'rcv1_category_2_labels.json'
        which_file = 'rcv1_category_2_labels'
    if which_data == 21:
        data_save_file = 'rcv1_category_2_nonover_labels.json'
        which_file = 'rcv1_category_2_nonover_labels'

    if which_data == 22:
        data_save_file = 'wos_topic_doc_delSci_withstop.json'
        which_file = 'wos_topic_doc_delSci_withstop'

    if which_data == 23:
        data_save_file = 'multi_lg_cade_brazilian_portuguese.json'
        which_file = 'multi_lg_portuguese'
    if which_data == 24:
        data_save_file = 'multi_lg_chinese.json'
        which_file = 'multi_lg_chinese'
    if which_data == 25:
        data_save_file = 'multi_lg_german.json'
        which_file = 'multi_lg_german'
    if which_data == 26:
        data_save_file = 'multi_lg_portuguese_dw_general_v2.json'
        which_file = 'multi_lg_portuguese_dw_v2'
    if which_data == 27:
        data_save_file = 'multi_lg_chinese_sogou.json'
        which_file = 'multi_lg_chinese_sogou'


    data_save_path_file = os.path.join(data_save_path, data_save_file)

    with open(data_save_path_file) as data_file:
        data_save_dic = json.load(data_file)
    topic_text_ten_largest_dic = data_save_dic

    K_lda = len(topic_text_ten_largest_dic.keys())

    # Save the raw data into list
    topic_list = []
    texts_list = []

    keys_list = list(topic_text_ten_largest_dic.keys())
    keys_list.sort()

    topic_no = 0
    for topic_tmp in keys_list:
        for id_tmp in range(len(topic_text_ten_largest_dic[topic_tmp])):
            topic_list.append(topic_no)
            texts_list.append(topic_text_ten_largest_dic[topic_tmp][id_tmp])
        topic_no += 1

    return K_lda, which_file, topic_list, texts_list


def get_topic_text_reuter_wordid_withstop(which_data, flag_fix_stopword=0, set_ps=1, generate_stop=None, flag_fix_length=0, set_docLength=20, limit_original_Length=None, ):
    '''
    Get the topic_list and the texts_list of reuters dataset. Shuffle the order but with stopwords.

    Input:
        - which_data: int, 1 or 0, 0: topic with more than 10 documents; 1: ten largest Topics.

        - flag_fix_length:
            1 or 0, flag to denote whether to fix the length of each document to a specific value;
            1 means we need to fix the length; 0 means no action will be taken.
        - set_docLength:
            the pre-setting of the length of each document: randomly choose set_docLength token from each doc without replacing
        - limit_original_Length:
            used to remove documents whose length is smaller than limit_original_Length.
            limit_original_Length should be larger than set_docLength.
            The default value is None, which means limit_original_Length is the same as set_docLength.
            However, when run the experiment for changing-lenght, limit_original_Length should be set as the upper limit of x-axis.

        - flag_fix_stopword:
            1 or 0, flag to denote whether to remove stopwords in the real corpus
            1 means remove; 0 means no action will be taken.
        - set_ps:
            pecentage of stopword remaining in the corpus.
            set_ps = 1 means no stopwords will be removed.
            set_ps = 0 means all stopwords will be removed.
        - generate_stop：
            Whether to use a specifically generated stopword list.
            None: use the default sotpword list for English from Mallet
            a float number: a parameter to generate stopword list
    Ouput:
        - K_lda: int, number of topics
        - which_file: str, '10largest' or '10more'
        - topic_list: list, list of topics in form of ID
        - texts_list: list, list of documents in form of ID

    Real corpus list:
        0: reuters_10more
        1: reuters_10largest
        2: reuters_top_odd10
        3: reuters_top_even10
        4: rcv1_1topic
        5: rcv1_1topic_v2

        6: t20NewsGroup_topic_doc_all_terms
        7: t20NewsGroup_topic_doc_no_short
        8: t20NewsGroup_topic_doc_no_stop
        9: t20NewsGroup_topic_doc_stemmed

        10: rcv1_category_2_labels_10pct
        11: rcv1_category_2_nonover_labels_20pct
        12: wos_topic_doc
        13: wos_topic_doc20length
        14: wos_topic_doc5length
        15: wos_topic_doc20length_random
        16: wos_topic_doc5length_random

        17: wos_topic_small_test_600doc

        20: rcv1_category_2_labels
        21: rcv1_category_2_nonover_labels
    '''

    ##########################################################
    # Get data from reuters
    ##########################################################

    data_save_path = os.path.abspath(os.path.join(os.pardir, 'data', 'reuters', 'data_filter'))

    if which_data == 0:
        data_save_file = 'reuters_filter_10more_dic.json'
        which_file = 'reuters_10more'
    if which_data == 1:
        data_save_file = 'reuters_filter_10largest_dic.json'
        which_file = 'reuters_10largest'
    if which_data == 2:
        data_save_file = 'reuters_filter_top_odd10_dic.json'
        which_file = 'reuters_top_odd10'
    if which_data == 3:
        data_save_file = 'reuters_filter_top_even10_dic.json'
        which_file = 'reuters_top_even10'
    if which_data == 4:
        data_save_file = 'rcv1_doc_oneTopic.json'
        which_file = 'rcv1_1topic'
    if which_data == 5:
        data_save_file = 'rcv1_doc_oneTopic_v2.json'
        which_file = 'rcv1_1topic_v2'

    if which_data == 6:
        data_save_file = 't20NewsGroup_topic_doc_all_terms.json'
        which_file = 't20NewsGroup_topic_doc_all_terms'
    if which_data == 7:
        data_save_file = 't20NewsGroup_topic_doc_no_short.json'
        which_file = 't20NewsGroup_topic_doc_no_short'
    if which_data == 8:
        data_save_file = 't20NewsGroup_topic_doc_no_stop.json'
        which_file = 't20NewsGroup_topic_doc_no_stop'
    if which_data == 9:
        data_save_file = 't20NewsGroup_topic_doc_stemmed.json'
        which_file = 't20NewsGroup_topic_doc_stemmed'

    if which_data == 10:
        data_save_file = 'rcv1_category_2_labels_10pct.json'
        which_file = 'rcv1_category_2_labels_10pct'
    if which_data == 11:
        data_save_file = 'rcv1_category_2_nonover_labels_20pct.json'
        which_file = 'rcv1_category_2_nonover_labels_20pct'
    if which_data == 12:
        data_save_file = 'wos_topic_doc.json'
        which_file = 'wos_topic_doc'
    if which_data == 13:
        data_save_file = 'wos_topic_doc20length.json'
        which_file = 'wos_topic_doc20length'
    if which_data == 14:
        data_save_file = 'wos_topic_doc5length.json'
        which_file = 'wos_topic_doc5length'

    if which_data == 15:
        data_save_file = 'wos_topic_doc20length_random.json'
        which_file = 'wos_topic_doc20length_random'
    if which_data == 16:
        data_save_file = 'wos_topic_doc5length_random.json'
        which_file = 'wos_topic_doc5length_random'

    if which_data == 17:
        data_save_file = 'wos_topic_90doc1topic_length80_small_test.json'
        which_file = 'wos_topic_small_test_600doc'

    if which_data == 18:
        data_save_file = 'wos_topic_doc_delSci.json'
        which_file = 'wos_topic_doc_delSci'
    if which_data == 19:
        data_save_file = 'wos_topic_doc20length_random_delSci.json'
        which_file = 'wos_topic_doc20length_random_delSci'

    if which_data == 20:
        data_save_file = 'rcv1_category_2_labels.json'
        which_file = 'rcv1_category_2_labels'
    if which_data == 21:
        data_save_file = 'rcv1_category_2_nonover_labels.json'
        which_file = 'rcv1_category_2_nonover_labels'

    data_save_path_file = os.path.join(data_save_path, data_save_file)

    with open(data_save_path_file) as data_file:
        data_save_dic = json.load(data_file)
    topic_text_ten_largest_dic = data_save_dic

    # if set the pecentage of stopwords, or not
    if flag_fix_stopword:
        topic_text_ten_largest_dic = {}
        topic_text_ten_largest_dic = get_real_corpus_fix_stopword(data_save_dic, data_save_file, set_ps, generate_stop=generate_stop)

    # if fix length or not
    if flag_fix_length:
        data_save_dic = {}
        data_save_dic = topic_text_ten_largest_dic

        topic_text_ten_largest_dic = {}
        topic_text_ten_largest_dic = set_doc_length_real_corpus(data_save_dic, set_docLength=set_docLength, limit_original_Length=limit_original_Length)

    K_lda = len(topic_text_ten_largest_dic.keys())

    # Save the raw data into list
    topic_list = []
    texts_list = []

    keys_list = list(topic_text_ten_largest_dic.keys())
    keys_list.sort()

    topic_name_no_dic = {}
    topic_no = 0
    for topic_tmp in keys_list:
        topic_name_no_dic[topic_no] = topic_tmp
        for id_tmp in range(len(topic_text_ten_largest_dic[topic_tmp])):
            topic_list.append(topic_no)
            texts_list.append(topic_text_ten_largest_dic[topic_tmp][id_tmp])
        topic_no += 1

    ##########################################################
    # Change the tokens into ids
    ##########################################################
    '''
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
    topic_list = list(topic_arr[arr])
    texts_list = list(texts_arr[arr])
    '''

    return K_lda, which_file, topic_list, texts_list


def set_doc_length_real_corpus(real_world_corpus_dic, set_docLength=20, limit_original_Length=None):
    '''
    Fix the length of each documents in the real corpora

    Input:
        - real_world_corpus_dic:
            corpus_dic = {
                'topic_1': [
                    ['t1_doc_1'],
                    ['t1_doc_2']
                    ...
                ],
                'topic_2': [
                    ['t2_doc_1'],
                    ['t2_doc_2']
                    ...
                ],
            }
        - set_docLength:
            the pre-setting of the length of each document: randomly choose set_docLength token from each doc without replacing
        - limit_original_Length:
            used to remove documents whose length is smaller than limit_original_Length.
            limit_original_Length should be larger than set_docLength.
            The default value is None, which means limit_original_Length is the same as set_docLength.
            However, when run the experiment for changing-lenght, limit_original_Length should be set as the upper limit of x-axis.

    Output:
            corpus_dic = {
                'topic_1': [
                    ['t1_doc_1_fix'],
                    ['t1_doc_2_fix']
                    ...
                ],
                'topic_2': [
                    ['t2_doc_1_fix'],
                    ['t2_doc_2_fix']
                    ...
                ],
            }

    '''

    if limit_original_Length is None:
        limit_original_Length = set_docLength
    if limit_original_Length < set_docLength:
        limit_original_Length = set_docLength

    key_list = list(real_world_corpus_dic.keys())
    key_list.sort()

    real_world_corpus_set_length_dic = {}
    for tmp_key in key_list:
        for tmp_doc in real_world_corpus_dic[tmp_key]:
            if len(tmp_doc) >= limit_original_Length:
                if tmp_key not in real_world_corpus_set_length_dic.keys():
                    real_world_corpus_set_length_dic[tmp_key] = []

                tmp_doc_fix_length = list(np.random.choice(tmp_doc, size=set_docLength, replace=0))
                real_world_corpus_set_length_dic[tmp_key] += [tmp_doc_fix_length]

    return real_world_corpus_set_length_dic


def get_stopword_list():

    stopword_data_path = os.path.abspath(os.path.join(os.pardir, 'data', 'stopword_list'))
    stopword_data_file = 'stopword_list_en'
    stopword_data_path_file = os.path.join(stopword_data_path, stopword_data_file)
    with open(stopword_data_path_file, 'r') as f:
        x = f.readlines()
    stopword_list = []
    for tmp_term in x:
        stopword_list += [tmp_term.split('\n')[0]]

    return stopword_list


def get_generated_stopword_list(data_save_file, generate_stop):

    stopword_data_path = os.path.abspath(os.path.join(os.pardir, 'data', 'stopword_list'))

    if generate_stop == -2:
        stopword_list = []

    elif generate_stop == -1:
        stopword_data_file = 'stopword_list_en'
        stopword_data_path_file = os.path.join(stopword_data_path, stopword_data_file)
        with open(stopword_data_path_file, 'r') as f:
            x = f.readlines()
        stopword_list = []
        for tmp_term in x:
            stopword_list += [tmp_term.split('\n')[0]]

    else:
        stopword_data_file = 'texts_filter-stops_stopwords_%s_Ns1000_pval-max0.0_dHbits-min%.1f' % (data_save_file.split('.')[0], generate_stop)
        stopword_data_path_file = os.path.join(stopword_data_path, stopword_data_file)
        with open(stopword_data_path_file, 'r') as f:
            x = f.readlines()
        stopword_list = []
        for tmp_term in x:
            stopword_list += [tmp_term.split('\n')[0]]

    return stopword_list


def get_real_corpus_fix_stopword(clean_data_dic, data_save_file, set_ps=1, generate_stop=None):
    '''
    Fix the stopwords in the real corpus into a pre-setting level, set_ps.
    Input:
        - set_ps:
            pecentage of stopword remaining in the corpus.
            set_ps = 1 means no stopwords will be removed.
            set_ps = 0 means all stopwords will be removed.
    '''
    if generate_stop is None:
        stopword_list = get_stopword_list()
    else:
        stopword_list = get_generated_stopword_list(data_save_file, generate_stop)

    clean_data_fix_stopword_dic = {}

    for tmp_key in clean_data_dic.keys():

        if tmp_key not in clean_data_fix_stopword_dic.keys():
            clean_data_fix_stopword_dic[tmp_key] = []

        for tmp_doc_old in clean_data_dic[tmp_key]:
            tmp_doc_new = []

            for tmp_token in tmp_doc_old:

                if tmp_token not in stopword_list:
                    tmp_doc_new += [tmp_token]
                else:
                    flag_del_stop = np.random.choice(2, 1, p=[set_ps, 1 - set_ps])
                    if flag_del_stop == 0:
                        tmp_doc_new += [tmp_token]

            clean_data_fix_stopword_dic[tmp_key] += [tmp_doc_new]

    return clean_data_fix_stopword_dic


'''
def get_topic_text_reuter_wordid_nostop(which_data):

    # Get the topic_list and the texts_list of reuters dataset. Shuffle the order but with stopwords.
    # Input:
    # - which_data: int, 1 or 0 ,
    #     0: topic with more than 10 documents;
    #     1: ten largest Topics.
    # Ouput:
    # - K_lda: int, number of topics
    # - which_file: str, '10largest' or '10more'
    # - topic_list: list, list of topics in form of ID
    # - texts_list: list, list of documents in form of ID

    ##########################################################
    # Get the stop words
    ##########################################################

    path_file_stopword = os.path.abspath(os.path.join(os.pardir, 'data', 'stopword_list_en'))
    with open(path_file_stopword, 'r') as f:
        x = f.readlines()
    all_data = np.array([np.array(h.split()).astype(np.str) for h in x])
    stopword_list = []
    for i in all_data:
        stopword_list.append(i[0])

    ##########################################################
    # Get data from reuters
    ##########################################################

    data_save_path = os.path.abspath(os.path.join(os.pardir, 'data', 'reuters', 'data_filter'))

    if which_data == 0:
        data_save_file = 'reuters_filter_10more_dic.json'
        which_file = 'reuters_10more'
    if which_data == 1:
        data_save_file = 'reuters_filter_10largest_dic.json'
        which_file = 'reuters_10largest'
    if which_data == 2:
        data_save_file = 'reuters_filter_top_odd10_dic.json'
        which_file = 'reuters_top_odd10'
    if which_data == 3:
        data_save_file = 'reuters_filter_top_even10_dic.json'
        which_file = 'reuters_top_even10'
    if which_data == 4:
        data_save_file = 'rcv1_doc_oneTopic.json'
        which_file = 'rcv1_1topic'
    if which_data == 5:
        data_save_file = 'rcv1_doc_oneTopic_v2.json'
        which_file = 'rcv1_1topic_v2'

    if which_data == 6:
        data_save_file = 't20NewsGroup_topic_doc_all_terms.json'
        which_file = 't20NewsGroup_topic_doc_all_terms'
    if which_data == 7:
        data_save_file = 't20NewsGroup_topic_doc_no_short.json'
        which_file = 't20NewsGroup_topic_doc_no_short'
    if which_data == 8:
        data_save_file = 't20NewsGroup_topic_doc_no_stop.json'
        which_file = 't20NewsGroup_topic_doc_no_stop'
    if which_data == 9:
        data_save_file = 't20NewsGroup_topic_doc_stemmed.json'
        which_file = 't20NewsGroup_topic_doc_stemmed'

    if which_data == 10:
        data_save_file = 'rcv1_category_2_labels_10pct.json'
        which_file = 'rcv1_category_2_labels_10pct'
    if which_data == 11:
        data_save_file = 'rcv1_category_2_nonover_labels_20pct.json'
        which_file = 'rcv1_category_2_nonover_labels_20pct'
    if which_data == 12:
        data_save_file = 'wos_topic_doc.json'
        which_file = 'wos_topic_doc'
    if which_data == 13:
        data_save_file = 'wos_topic_doc20length.json'
        which_file = 'wos_topic_doc20length'
    if which_data == 14:
        data_save_file = 'wos_topic_doc5length.json'
        which_file = 'wos_topic_doc5length'

    if which_data == 20:
        data_save_file = 'rcv1_category_2_labels.json'
        which_file = 'rcv1_category_2_labels'
    if which_data == 21:
        data_save_file = 'rcv1_category_2_nonover_labels.json'
        which_file = 'rcv1_category_2_nonover_labels'

    data_save_path_file = os.path.join(data_save_path, data_save_file)

    with open(data_save_path_file) as data_file:
        topic_text_ten_largest_dic = json.load(data_file)

    K_lda = len(topic_text_ten_largest_dic.keys())

    # Save the raw data into list

    topic_list = []
    texts_list = []

    keys_list = list(topic_text_ten_largest_dic.keys())
    keys_list.sort()

    topic_name_no_dic = {}
    topic_no = 0
    for topic_tmp in keys_list:
        topic_name_no_dic[topic_no] = topic_tmp
        for id_tmp in range(len(topic_text_ten_largest_dic[topic_tmp])):
            topic_list.append(topic_no)
            texts_list.append(topic_text_ten_largest_dic[topic_tmp][id_tmp])
        topic_no += 1

    ##########################################################
    # Change the tokens into word_id, and get gid of the stop words
    ##########################################################

    # save the corpora with words
    texts_list_word = copy.deepcopy(texts_list)

    texts_list_word_tmp = copy.deepcopy(texts_list_word)
    dict_gs = corpora.Dictionary(texts_list_word_tmp)
    new_dict_token2id = dict(dict_gs.token2id)

    texts_list_id_tmp = []

    total_word_num = 0
    nostop_word_num = 0

    for document_id in range(len(texts_list_word_tmp)):
        texts_list_id_tmp.append([])
        for token_id in range(len(texts_list_word_tmp[document_id])):
            total_word_num += 1
            token_tmp = texts_list_word_tmp[document_id][token_id]
            if token_tmp not in stopword_list:
                nostop_word_num += 1
                texts_list_id_tmp[document_id].append(str(new_dict_token2id[token_tmp]))

    # Save all the ids back to oright list
    texts_list = copy.deepcopy(texts_list_id_tmp)

    # Shuffle texts_list and topic_list
    topic_arr = np.array(topic_list)
    texts_arr = np.array(texts_list)
    arr = np.arange(len(topic_list))
    np.random.shuffle(arr)
    topic_list = list(topic_arr[arr])
    texts_list = list(texts_arr[arr])

    return K_lda, which_file, topic_list, texts_list
'''
