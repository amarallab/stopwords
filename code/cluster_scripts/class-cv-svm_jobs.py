#!/usr/bin/env python
'''
Job submission for quest
'''
#This import is necessary so that we can modify the permissions on the
#submission script
import os
import numpy as np
from itertools import product


## Upperlimit for running time: unit is hour
walltime = 4

## Specific marker for folders belong to the same batch of jobs
program_name = 'class-cv-svm_code.py'
current_path = os.path.abspath(os.path.join(''))
oe_path = 'log/'
## parameters for loading data

folder_marker = 'class-cv-svm_20190607'

# ## set up data
list_name_corpus = [
        'reuters_filter_10largest_dic',
        't20NewsGroup_topic_doc_no_short',
        'wos_topic_doc_delSci_withstop',
        'multi_lg_german',
        'multi_lg_portuguese_dw_general_v2',
        'multi_lg_chinese',
        'multi_lg_chinese_sogou',
        ]

list_name_method = [
'INFOR',
'BOTTOM',
'TOP',
'TFIDF',
'TFIDF_r',
'MANUAL',
'TOP-BOTTOM'
            ]

path_data = os.path.join(os.pardir,os.pardir,'data','s2019-05-29_stopword-corpora')

i_job = 0
for name_corpus,name_method in product(
        list_name_corpus,
        list_name_method,
    ):
    
    ## define language for stopword list
    lang = 'en' ## default
    if 'german' in name_corpus:
        lang = 'de'
    if 'portuguese' in name_corpus:
        lang = 'pt'
    if 'chinese' in name_corpus:
        lang = 'cn' 

    jobscript = os.path.join("bash","%s_%s.sh"%(folder_marker,i_job) )
    queue_out = open(jobscript,"w") 
    queue_out.write("""#!/bin/bash
#SBATCH -A p30656
#SBATCH -p short
#SBATCH -t %s:00:00
#SBATCH -N 1
#SBATCH --job-name sample_jobscript

pwd
cd %s
module load python/anaconda3.6
source activate python3
time python %s %s %s %s %s %s
                    """%( walltime,\
                          current_path, \
                          str(program_name), name_corpus, name_method, folder_marker, lang, path_data) )

    queue_out.close()
    os.system("sbatch %s"%(jobscript))
    i_job += 1 
    print(i_job)

            


