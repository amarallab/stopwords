import numpy as np
from collections import Counter
from common.convert_states import *
from scipy.optimize import linear_sum_assignment
from random import shuffle


def state_dwz_nmi(state_dwz1_, state_dwz2_, K1_, K2_, normalized=True):
    '''
    Calculate the normalized mutual information (NMI) between two labeled dwz-states, i.e. how much the two labeled states overlap.
    In: - state_dwz1_,
        - state_dwz2_,
        - K1_, # of topics in state_dwz1_, int
        - K2_, # of topics in state_dwz2_, int
    Out:
        - NMI, float
    '''
    # VI_ = 0.0
    N_ = len(state_dwz1_)
    n_tt_ = np.zeros((K1_, K2_))

    state_dwz1_s, state_dwz1_ = state_perturb_wd(state_dwz1_)  # sort and shuffle labels across same words and docs
    state_dwz2_s, state_dwz2_ = state_perturb_wd(state_dwz2_)  # sort and shuffle labels across same words and docs

    list_z1_z2_ = [(state_dwz1_[i_][2], state_dwz2_[i_][2]) for i_ in range(N_)]
    c_z1_z2_ = Counter(list_z1_z2_)
    for z1_z2_, n_z1_z2_ in c_z1_z2_.items():
        n_tt_[z1_z2_[0], z1_z2_[1]] += n_z1_z2_
    p_tt_ = n_tt_ / float(N_)
    p_t1_ = np.sum(p_tt_, axis=1)
    p_t2_ = np.sum(p_tt_, axis=0)
    H1_ = sum([-p_ * np.log(p_) for p_ in p_t1_ if p_ > 0.0])
    H2_ = sum([-p_ * np.log(p_) for p_ in p_t2_ if p_ > 0.0])
    MI_ = 0.0
    for i_ in range(K1_):
        for j_ in range(K2_):
            p1_ = p_t1_[i_]
            p2_ = p_t2_[j_]
            p12_ = p_tt_[i_, j_]
            if p12_ > 0.0:
                MI_ += p12_ * np.log(p12_ / (p1_ * p2_))
    if normalized is True:
        NMI_ = 2.0 * MI_ / (H1_ + H2_)
    else:
        NMI_ = 1.0 * MI_
        # if we want to return the unnormalized mutual information
    return NMI_