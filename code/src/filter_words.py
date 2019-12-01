import os, sys
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

from data_io import texts_nwd_csr


def nwd_H_J_w_csr(n_wd):
    '''
    Calculate the conditional entropy H({d}|w) quantifying the importance of each word.
    ADAPTED VESION of nwd_H_J_w for csr-sparse matrices.
    IN:
    - n_wd, scipy-csr-matrix, shape=VxD (V=number of words, D=number of docs)
    OUT:
    - H_J_w, np.array, shape=V
    '''
    V,D=n_wd.shape

    ## counts of each word as np-array
    N_w = np.array(n_wd.sum(axis=1).transpose())[0]

    ## data at non-zero elemnts
    n = n_wd.data
    ## indices at non-zero elemnts
    row_n,col_n=n_wd.nonzero()

    ## entropy-log for non-zero data [BITS!]
    n_H = n*np.log2(n)
    ## we only keep data for non-zero entropies
    ind_sel = np.where(n_H>0)

    ## construct new sparse matrix with entropy values
    row_H = row_n[ind_sel]
    col_H = col_n[ind_sel]
    data_H = n_H[ind_sel]
    X_H = csr_matrix((data_H, (row_H, col_H)), shape=(V, D))

    ## calculate the proper entropy
    H_J_w = -np.array(X_H.sum(axis=1).transpose())[0]/N_w + np.log2(N_w)
    return H_J_w

### SHUFFLING NULL MODEL

def nwd_H_shuffle(
    n_wd,
    N_s=10**2,
    ):
    '''
    Calculate the empirical and expected conditional entropy \tilde{H({d}|w)} using random null.
    The random null model shuffles word tokens across documents preserving marginal counts n(w) and n(d)
    IN:
    - n_wd, sparse matrix of number of counts of word w in doc d
    optional:
    - N_s, int (default: 100); number of random realizations

    OUT:
    - dictionary with keys
        - H-emp; empirical entropy
        - H-null-mu; average of entropy from random null model over N_s realizations
        - H-null-std; standard dev of entropy from random null model over N_s realizations
        - F-emp; empirical rel frequency of each word
        - params; parameter values
    '''
    ## get:
    ## - cpunts of each document N_d
    ## - emp frequencies F_w
    ## - estimation of conditional entopy
    V,D=n_wd.shape
    N_w = np.array(n_wd.sum(axis=1).transpose())[0]
    N_d = np.array(n_wd.sum(axis=0))[0]
    N = np.sum(N_d)
    F_w = N_w/float(N)

    #### empirical entropy
    arr_H_w = nwd_H_J_w_csr(n_wd)

    #### calculate expected entropy (and std)
    arr_Hmu_w = np.zeros(V)
    arr_H2mu_w = np.zeros(V)


    for i_s in range(N_s):
        n_wd_rand = nwd_csr_shuffle(n_wd)
        arr_H_w_rand = nwd_H_J_w_csr(n_wd_rand)
        # update average
        arr_Hmu_w += arr_H_w_rand/float(N_s)
        # update squared average
        arr_H2mu_w += arr_H_w_rand**2/float(N_s)

    ## get standard dev
    ## avoid nan if E[X^2]-E[X]^2~ -10**(-14) --> set to 0 
    s_tmp = arr_H2mu_w - arr_Hmu_w**2 
    arr_Hsigma_w = np.sqrt(  0.0*(s_tmp<0) + s_tmp*(s_tmp>=0.) )

    result = {}
    result['H-emp'] = arr_H_w
    result['H-null-mu'] = arr_Hmu_w
    result['H-null-std'] = arr_Hsigma_w
    result['F-emp'] = F_w
    result['params'] = {
        'N_s':N_s,
        }, 

    return result


def nwd_tfidf_csr(n_wd_csr):
    '''tfidf
    Set result to 0 if nan
    '''
    V,D = n_wd_csr.shape

    ## term frequecny
    tf_w = np.array(n_wd_csr.sum(axis=1))[:,0]
    ## document frequency
    rows,cols=n_wd_csr.nonzero()
    data = [1]*len(rows)
    df_wd_csr = csr_matrix( ( data, (rows, cols) ), shape=(V, D), dtype=np.int64, copy=False)
    df_w = np.array(df_wd_csr.sum(axis=1))[:,0]

    ## tfidf
    tfidf_w = tf_w/df_w * np.log( D/df_w )
    return tfidf_w

def nwd_csr_shuffle(n_wd_csr):
    '''
    Obtain n_wd from shuffling tokens across documents.
    Gives n_wd from one random realization
    '''
    N_w = np.array(n_wd_csr.sum(axis=1).transpose())[0]
    N_d = np.array(n_wd_csr.sum(axis=0))[0]

    list_texts_flat = []
    for i_w,n_w in enumerate(N_w):
        list_texts_flat += [i_w]*n_w
    np.random.shuffle(list_texts_flat)

    list_texts_random = []
    n=0
    for m in N_d:
        text_tmp = list_texts_flat[n:n+m]
        list_texts_random+=[text_tmp]
        n+=m

    ## this is the current bottleneck: takes 6 times longer than the shuffling
    ## given the row,col,data the csr-constructor takes all the time
    n_wd_csr_r, dict_w_iw_r = texts_nwd_csr(list_texts_random)
    return n_wd_csr_r

def run_stopword_statistics(list_texts, N_s = 100, path_stopword_list = None):
    '''
    Make a dataframe with ranking of words according to different metrics
    '''

    ## make csr matrix
    n_wd_csr, dict_w_iw = texts_nwd_csr(list_texts)
    V,D = n_wd_csr.shape

    ## get entropy measure
    result_H = nwd_H_shuffle(n_wd_csr,N_s=N_s)

    ## get tfidf
    arr_tfidf_w = nwd_tfidf_csr(n_wd_csr)

    ## make dataframe
    df=pd.DataFrame(index = sorted(list(dict_w_iw.keys())) )

    df['F'] = result_H['F-emp']
    df['I'] = result_H['H-null-mu'] - result_H['H-emp']
    df['tfidf'] = arr_tfidf_w

    ## get stopword list if file with list was provided
    if path_stopword_list != None:
        with open(path_stopword_list,'r') as f:
            x = f.readlines()
        stopwords = [h.strip() for h in x]
        arr_manual = np.zeros(V)
        for w in stopwords:
            try:
                iw = dict_w_iw[w]
                arr_manual[iw] = 1
            except KeyError:
                pass
        df['manual'] = arr_manual
        ## replace 0 by nan; such that these words will not be filtered
        df['manual']=df['manual'].replace(to_replace=0,value=np.nan)


    ## get entropy and random entropy too
    df['H'] = result_H['H-emp']
    df['H-tilde'] =  result_H['H-null-mu']
    df['H-tilde_std'] =  result_H['H-null-std']
    df['N'] = np.array(n_wd_csr.sum(axis=1))[:,0] ## number of counts

    return df

def make_stopwords_filter(
    df,
    method = 'INFOR', 
    cutoff_type = 'p', 
    cutoff_val = 0., 
                        ):
    '''
    Create filter from stopword-statistics dataframe.
    IN:
    - df with stopword statistics; get from filter_stopwords.run_stopword_statistics()
    
    Options for filtering.
    - method [defines the observable S on which to filter]. 
    We filter words with low values of S.
        - INFOR, filter words with low values of Information-content I [S=I]
        - INFOR_r,  filter words with high values of Information-content I [S=-I]
        - BOTTOM, filter words with low values of frequency [S = F]
        - TOP, filter words with high values of frequency [S = 1/F]
        - TFIDF, filter words with low values of tfidf [S=tfidf]
        - TFIDF_r, filter words with high values of tfidf [S=-tfidf]
        - MANUAL, filter words from manual stopword list; supply path via path_stopword_list 
        S = 1 if word is in the list, else it is nan (cannot be considered for removal)
        - RANDOM, filter words randomly, [S=1 for all words]; yields alphabetic ordering
        
    - cutoff_type [defines the way in which we choose the cutoff]
        - p, selects stopword list such that a fraction p of tokens gets removed (approximately)
        - n, selects stopword list such that a number n of types gets removed
        - t, selects stopword list such that all words with S<=S_t get removed
    
    - cutoff_val [defines the value on which to do the thresholding, see cutoff_type for details]
    
    Note that in case of ties in the observable S (e.g. S=1 in the case of manual), words are ordered alphabetically.
    '''
    ## choose obserable for filteting
    if method == 'INFOR':
        ## filter low I-values
        S = df['I']
    elif method == 'INFOR_r':
        ## filter large I-values (multiply by -1; cannot take inverse since we have I=0)
        S = -1.0*df['I']
    elif method == 'BOTTOM':
        ## filter low frequency
        S = df['N']
    elif method == 'TOP':
        ## filter high frequency (take inverse)
        S = 1./df['N']
    elif method == 'TFIDF':
        ## filter low values of tfidf
        S = df['tfidf']
    elif method == 'TFIDF_r':
        ## filter large values of tfidf (multiply by -1; can we have tfidf = 0?)
        S = -df['tfidf']
    elif method == 'MANUAL':
        S = df['manual']
    elif method == 'RANDOM':
        ## filter words 'randomly'; note that there is an alphabetic ordering below
        ## we assign the same value to each word
        ## note this makes only sense for types 'p' and 'n' (no threshold possible)
        S = 1.0+0.0*df['F']
    else:
        print('You did not choose a valid filtering method')
        
    ## choose cutoff method
    S = S.dropna().sort_index().sort_values(kind='mergesort')
    ## this does the folowing things
    ## - drop nan-entries == words that should not be filtered
    ## - sort according to the variable indicated
    ## - if there is a tie, sort according to index -- alphabetically
    
    ## make the dataframe for filtering
    ## S is the filtering observable
    ## F-cumsum is used to filter an approximate fraction of tokens from the corpus
    df_filter = pd.DataFrame(index = S.index )
    df_filter['F-cumsum']=df.loc[S.index]['F'].cumsum()
    df_filter['S'] = S
    
    if cutoff_type == 'p':
    ## filter a fraction of tokens
        S_p = cutoff_val
        df_filter = df_filter[df_filter['F-cumsum']<=S_p]
    elif cutoff_type == 'n':
    ## filter a number of types
        S_n = cutoff_val
        df_filter = df_filter.iloc[:S_n]
    elif cutoff_type == 't':
    ## filter at a threshld value of the method-observable S
        S_t = cutoff_val
        df_filter = df_filter.loc[S[S<S_t].index]
    else:
        print('You did not choose a proper cutoff method')
        
    return df_filter

def make_stopwordlist(df,
    method = 'INFOR', 
    cutoff_type = 'p', 
    cutoff_val = 0., ):
    '''
    Return a list of stopwords to be filtered according to different heuristics.
    see make_stopwords_filter for details
    - method
        - INFOR, filter words with low values of Information-content I [S=I]
        - INFOR_r,  filter words with high values of Information-content I [S=-I]
        - BOTTOM, filter words with low values of frequency [S = F]
        - TOP, filter words with high values of frequency [S = 1/F]
        - TFIDF, filter words with low values of tfidf [S=tfidf]
        - TFIDF_r, filter words with high values of tfidf [S=-tfidf]
        - MANUAL, filter words from manual stopword list; supply path via path_stopword_list 
        - TOP-BOTTOM, combine Top with Bottom (equal proportion in terms of tokens)
        - RANDOM, selects stopwords randomly (actually alphabetically)

    - cutoff_type [defines the way in which we choose the cutoff]
        - p, selects stopword list such that a fraction p of tokens gets removed (approximately)
        - n, selects stopword list such that a number n of types gets removed
        - t, selects stopword list such that all words with S<=S_t get removed
    
    - cutoff_val [defines the value on which to do the thresholding, see cutoff_type for details]
    '''
    
    list_stopwords = []

    if method in ['INFOR','TOP','BOTTOM','TFIDF','TFIDF_r','MANUAL','RANDOM']:
        df_filter = make_stopwords_filter(df,
                                  method = method,
                                  cutoff_type = cutoff_type, 
                                  cutoff_val = cutoff_val, )
        list_stopwords += list(df_filter.index)
    elif method in ['TOP-BOTTOM']:
    ##remove top AND bottom words
    ## only works for cutoff_type = 'p'
    ## remove approx same amount of tokens from top and bottom
        cutoff_type_tmp = 'p'

        method_tmp = 'TOP'
        cutoff_val_tmp = 0.5*cutoff_val
        df_filter_tmp = make_stopwords_filter(df,
                                      method = method_tmp,
                                      cutoff_type = cutoff_type_tmp, 
                                      cutoff_val = cutoff_val_tmp, )
        list_stopwords += list(df_filter_tmp.index)
        method_tmp = 'BOTTOM'
        cutoff_val_tmp = 0.5*cutoff_val
        df_filter_tmp = make_stopwords_filter(df,
                                      method = method_tmp,
                                      cutoff_type = cutoff_type_tmp, 
                                      cutoff_val = cutoff_val_tmp, )
        list_stopwords += list(df_filter_tmp.index)
    else:
        print('You did not choose a proper cutoff method')
    return list_stopwords

def remove_stopwords_from_list_texts(list_texts, list_words_filter):
    '''
    remove list of stopwords from a list of texts

    Returns list of texts; in which every type from list_words_filter is removed.

    Note: can yield 'empty' documnts
    '''
    set_words_filter = set(list_words_filter)
    list_texts_filter = [ [h for h in doc if h not in set_words_filter] for doc in list_texts ]
    return list_texts_filter