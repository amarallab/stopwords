# stopwords

Code for the paper:

M. Gerlach, H. Shi, L.A.N. Amaral: "A universal information theoretic approach to the identification of stopwords" (2019)


## Structure

- data.  
  Contains the different language and gene datasets
- code
  - cluster_output  
  Output files from running bulk analysis of topic models and document classification in `cluster_scripts/`
  - cluster_scripts  
  Scripts run on [Quest](https://www.it.northwestern.edu/research/user-services/quest/) to do bulk analysis for topic models and document classification.  submitted `*_jobs.py`. requires running the stopword-statistics beforehand (see `scripts/`). also requires installation of the different topic-models (see below).
  - figures  
  figures from final analysis. run notebooks in `figures_notebooks/`
  - figures_notebooks
  create the final figures contained in the manuscript.
  - output  
  Output files from running the stopword statistics in `scripts/`
  - src  
  internal source-files
  - src_tm  
  external source-files from topic models used in the analysis
  - tmp  
  temporary folder needed to store intermediate results when running topic models.

## installation steps

- install required packages
  - python (3.6.3)
  - gensim (0.13.1) for ldavb topic model
  - pandas (0.24.2)
  - jupyter (1.0.0)
  - scikit-learn (0.21.2)
- configure the topic models (code/src_tmp)
  - mallet  
    `$ cd code/src_tm/external/mallet-2.0.8RC3`  
    `$ ant`
  - hdp  
    `cd code/src_tm/external/hdp-bleilab/hdp-faster`  
    `make`

## running

- Calculating stopword statistics of each corpus  
  run the files in `code/scripts/script_run-stopword-statistics_*.py`. this will save statistics on stopwords (I, tfidf, ...) for each corpus in `code/output`
- Run topic models and document classification  
  I submitted the jobs running the bulk analysis using the scripts in `code/cluster_scripts/*_jobs.py` to the cluster; this likely needs to be adapted if you want to run it somewhere else. The corresponding code is in  `code/cluster_scripts/*_code.py`. The results are saved in `code/cluster_output`.
- make the figures  
  The notebooks in `code/figures_notebooks/*.ipynb` use the results from the previous steps to generate the figures. they will be saved in `code/figures/` 
