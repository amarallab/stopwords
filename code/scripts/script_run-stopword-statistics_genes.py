import os,sys
import numpy as np
import pandas as pd

## save the stopword statistics for all corpora

src_dir = os.path.join(os.pardir, 'src')
sys.path.append(src_dir)

from filter_words import run_stopword_statistics
from data_io import texts_nwd_csr


path_read = os.path.join(os.pardir,os.pardir,'data','data-rna_2019-05-30')
fname_read = 'PBMC_1k_RNAseq_rawcount.csv'
filename = os.path.join(path_read,fname_read)
df_wd = pd.read_csv(filename,index_col=0,na_values=0).dropna(how='all',axis=0).to_sparse() ## contains the data


from filter_words import *

N_s = 1000
## make csr matrix
n_wd_csr = csr_matrix(df_wd.to_coo()).astype('int')
V,D = n_wd_csr.shape

## get entropy measure
result_H = nwd_H_shuffle(n_wd_csr,N_s=N_s)

## get tfidf
arr_tfidf_w = nwd_tfidf_csr(n_wd_csr)


## make dataframe
df=pd.DataFrame(index = df_wd.index )

df['F'] = result_H['F-emp']
df['I'] = result_H['H-null-mu'] - result_H['H-emp']
df['tfidf'] = arr_tfidf_w

## get entropy and random entropy too
df['H'] = result_H['H-emp']
df['H-tilde'] =  result_H['H-null-mu']
df['H-tilde_std'] =  result_H['H-null-std']
df['N'] = np.array(n_wd_csr.sum(axis=1))[:,0] ## number of counts

path_save = '../output'
fname_save = 'PBMC_1k_RNAseq_rawcount_stopword-statistics_Ns%s.csv'%(N_s)
filename = os.path.join(path_save,fname_save)
df.to_csv(filename)
