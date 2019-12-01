import os, sys, pickle, random
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

src_dir = os.path.join(os.pardir, 'src')
sys.path.append(src_dir)

from filter_words import run_stopword_statistics
from filter_words import make_stopwords_filter
from filter_words import make_stopwordlist
# from filter_words import remove_stopwords_from_list_texts
from shi_json import shi_json_to_texts
from data_io import texts_nwd_csr
from classification import classification_cv_svm

#####################################################################################################
## passing an argument
name_corpus = sys.argv[1]
method = sys.argv[2]
folder_marker = sys.argv[3]
lang = sys.argv[4] ## cn, de, en, pt
path_data = sys.argv[5]



## Load data
# path_read = '../../data/'
fname_read = '%s.json'%(name_corpus)
filename = os.path.join(path_data,fname_read)


## read hanyus json-format
x_data = shi_json_to_texts(filename) ## contains the data
## vocabulary and lsit of texts
list_words = x_data['list_w']
list_texts = x_data['list_texts']
dict_w_iw = x_data['dict_w_iw']
list_labels = x_data['list_c']
D = len(list_texts)

## sparse dataframe
X_csr,dict_w_iw =texts_nwd_csr(list_texts)
df = pd.SparseDataFrame(X_csr.transpose(),index=np.arange(D),columns=list_words)


## setup the stopwords stats
## run the stopwords statistics
path_stopword_list = os.path.join(path_data,'stopword_list_%s'%(lang))  ## path to stopword list
N_s = 100 ## number of realizations
df_stop = run_stopword_statistics(list_texts,N_s=N_s,path_stopword_list=path_stopword_list)

## loop over fraction of tokens tormeove
cutoff_type = 'p'


## set lower upper limit
pmin = 0.0
pmax = 0.95
if method=='MANUAL':
    ###get maximum fraction fr manual
    method_tmp = 'MANUAL'
    cutoff_type_tmp = 'p'
    cutoff_val_tmp = 1.

    df_filter_tmp = make_stopwords_filter(df_stop,
                                  method = method_tmp,
                                  cutoff_type = cutoff_type_tmp, 
                                  cutoff_val = cutoff_val_tmp, )
    pmax = df_filter_tmp['F-cumsum'].values[-1]

arr_p = np.linspace(pmin,pmax,20)

kfold=10
arr_acc = np.zeros(( kfold, len(arr_p) ))
for i_p,p in enumerate(arr_p):
    cutoff_val = p
    list_stopwords = make_stopwordlist(df_stop,
                                  method = method,
                                  cutoff_type = cutoff_type, 
                                  cutoff_val = cutoff_val, )
    df_sel = df.drop(labels = list_stopwords,axis=1)
    X = df_sel.to_coo().astype('int').tocsr()
    Y = np.array(list_labels)
    result = classification_cv_svm(X,Y,kfold=kfold)
    
    arr_acc[:,i_p] = result['acc']


## make an output path
path_save = os.path.join(os.pardir,'cluster_output','%s'%(folder_marker))
if os.path.exists(path_save):
    pass
else:
    os.mkdir(path_save)

    
# ## save the result

fname_save = '%s_%s_acc.npz'%(name_corpus,method)
filename = os.path.join(path_save,fname_save)
np.savez(filename,p=arr_p,acc = arr_acc)

## only saves the accuracy but we would also like the p-array since its different
# fname_save = '%s_%s_acc.npy'%(name_corpus,method)
# filename = os.path.join(path_save,fname_save)
# np.savetxt(filename,arr_acc)
