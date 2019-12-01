import numpy as np
from collections import Counter


from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_mutual_info_score

def classification_cv_svm(X, Y, kfold=10, rd_seed = 42):
    '''
    Perform kfold CV for the SVM.
    X is a feature matrix (samples x features); can be sparse (csr)
    Y is a list of labels
    return a dictionary with different evaluation stats
    '''
    kf = KFold(n_splits=kfold,random_state=rd_seed,shuffle=True)

    list_acc = []
    list_ami = []
    i_k = 0 ## index for the kth split
    for train_index, test_index in kf.split(X):
        i_k += 1

        ## data
        X_train = X[train_index]
        X_test = X[test_index]
        ## labels
        Y_train = Y[train_index]
        Y_test = Y[test_index]


        ## fit model and predict
        model = LinearSVC()
        model.fit(X_train,Y_train)
        Y_pred = model.predict(X_test)
        
        ## evaluate
        acc = accuracy_score(Y_test,Y_pred)
        list_acc += [acc]

        ami = adjusted_mutual_info_score(Y_test,Y_pred)
        list_ami += [ami]

    result = {}
    result['acc'] = list_acc
    result['ami'] = list_ami
    return result

def classification_cv_nb(X, Y, kfold=10, rd_seed = 42):
    '''
    Perform kfold CV for the SVM.
    X is a feature matrix (samples x features); can be sparse (csr)
    Y is a list of labels
    return a dictionary with different evaluation stats
    '''
    kf = KFold(n_splits=kfold,random_state=rd_seed,shuffle=True)

    list_acc = []
    list_ami = []
    i_k = 0 ## index for the kth split
    for train_index, test_index in kf.split(X):
        i_k += 1

        ## data
        X_train = X[train_index]
        X_test = X[test_index]
        ## labels
        Y_train = Y[train_index]
        Y_test = Y[test_index]


        ## fit model and predict
        model = MultinomialNB()
        model.fit(X_train,Y_train)
        Y_pred = model.predict(X_test)
        
        ## evaluate
        acc = accuracy_score(Y_test,Y_pred)
        list_acc += [acc]

        ami = adjusted_mutual_info_score(Y_test,Y_pred)
        list_ami += [ami]

    result = {}
    result['acc'] = list_acc
    result['ami'] = list_ami
    return result