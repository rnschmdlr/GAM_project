# %%
from scipy.stats.contingency import crosstab
import numpy as np

def direct_transfer_entropy(X, Y):
    N, S = X.shape # N = number of timesteps, S = number of slices
    TE = 0
    JPD_tgt_src = np.zeros((2, 2)) # target and source
    JPD_tgt_tgtp = np.zeros((2, 2)) # target and target past
    JPD_src_tgtp = np.zeros((2, 2)) # source and target past
    JPD_tgt_src_tgtp = np.zeros((2, 2, 2)) # target, source, target past == probability mass function

    # loop through all of the time series once to fill out the histograms for the PDF 
    for t in range(1, N):
        # calculate joint probability distributions (JPD)
        JPD_tgt_src += crosstab(Y[t], X[t-1]).count / N 
        JPD_tgt_tgtp += crosstab(Y[t], Y[t-1]).count / N 
        JPD_src_tgtp += crosstab(X[t-1], Y[t-1]).count / N 
        JPD_tgt_src_tgtp += crosstab(Y[t], X[t-1], Y[t-1]).count / N 

    # calculate conditional dependencies
    P_tgt_cond_src_and_tgtp = JPD_tgt_src_tgtp / JPD_src_tgtp
    P_tgt_cond_tgtp = JPD_tgt_tgtp / JPD_tgt_tgtp.sum(axis=0)
    
    #sum over all of the possible states to compute the average log ratio
    for t in range(2): # t = target
        for s in range(2): # s = source
            for tp in range (2): # tp = target past
                if JPD_tgt_src_tgtp[t,s,tp] > 0 and P_tgt_cond_src_and_tgtp[t,s,tp] > 0 and P_tgt_cond_tgtp[t,tp] > 0:
                    TE += JPD_tgt_src_tgtp[t,s,tp] * np.log2(P_tgt_cond_src_and_tgtp[t,s,tp] / P_tgt_cond_tgtp[t,tp])
    return TE
            

def pairwise_direct_transfer_entropy(seg_mat):
    n_tsteps, n_slice, n_loci = seg_mat.shape
    te_mat = np.zeros(shape=(n_loci, n_loci))

    for loc1 in range(n_loci):
        for loc2 in range(n_loci):
            if loc1 != loc2:
                te_mat[loc1,loc2] = direct_transfer_entropy(seg_mat[:,:,loc1], seg_mat[:,:,loc2])

    return te_mat



# %%
'''Test function'''
# generate data
n_loci = 15
n_tsteps = 50
n_slice = 1000
seg_mat = np.random.randint(2, size=(n_tsteps, n_slice, n_loci)) # pseudo co-segregation matrix

pairwise_direct_transfer_entropy(seg_mat)


