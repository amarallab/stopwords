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
walltime = 48
mem_per_cpu_mb = 10*1024

## Specific marker for folders belong to the same batch of jobs
program_name = 'tm-run_code.py'
current_path = os.path.abspath(os.path.join(''))
oe_path = 'log/'
## parameters for loading data

folder_marker = 'tm-run_20190620'

# ## set up data
list_corpus_name = [
        'reuters_filter_10largest_dic',
        't20NewsGroup_topic_doc_no_short',
        'wos_topic_doc_delSci_withstop',
        'multi_lg_german',
        'multi_lg_portuguese_dw_general_v2',
        'multi_lg_chinese_sogou',
        ]

list_method = [
'INFOR',
'TFIDF',
'MANUAL',
'BOTTOM',
'TOP',
'TOP-BOTTOM',
'RANDOM',
            ]

list_topic_model = [
    'hdp',
    'ldags',
    'ldavb'
    ]

N_s = 1000
k_set = 20
n_rep = 10

path_save = os.path.join(os.pardir,'cluster_output','%s'%(folder_marker))
if os.path.exists(path_save):
    pass
else:
    os.mkdir(path_save)
path_log = os.path.join(path_save,'log')
if os.path.exists(path_log):
    pass
else:
    os.mkdir(path_log)

i_job = 0
for corpus_name,method, topic_model in product(
        list_corpus_name,
        list_method,
        list_topic_model
    ):
    oe_file = os.path.join(path_log,'%s.out'%(i_job))

    jobscript = os.path.join("bash","%s_%s.sh"%(folder_marker,i_job) )
    queue_out = open(jobscript,"w") 
    queue_out.write("""#!/bin/bash
#SBATCH -A p30656
#SBATCH -p normal
#SBATCH -t %s:00:00
#SBATCH -N 1
#SBATCH --job-name sample_jobscript
#SBATCH --output="%s"
#SBATCH --mem-per-cpu=%s
pwd
cd %s
module load python/anaconda3.6
source activate stopwords
time python %s %s %s %s %s %s %s %s
                    """%( walltime, oe_file, mem_per_cpu_mb, \
                          current_path, \
                          str(program_name), corpus_name, N_s, method, topic_model, n_rep,path_save, k_set) )

    queue_out.close()
    os.system("sbatch %s"%(jobscript))
    i_job += 1 
    print(i_job)

            


