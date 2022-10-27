import numpy as np

def transfer_entropy(pXY, pyYX, pyY):
    """Returns the transfer entropy of two probability distributions as defined by Schreiber et al. using KLD"""

    # where p(YY) is 0, p(XY) is implied to be 0 and the contribution of the log term is interpreted as 0 
    pq_ratio = np.divide(pyYX, pyY, out=np.zeros_like(pyYX), where=pyY!=0)

    # result is zero if pXY is 0 because lim_x->0+ xlog(x) = 0
    te = np.sum(pXY * np.sum(pyYX * np.log2(pq_ratio, out=np.zeros_like(pq_ratio), where=pq_ratio!=0), axis=1), axis=0)

    return te


def multivariate_transfer_entropy(transition_pmat):
    n_loci = transition_pmat.shape[0]
    n_tsteps = transition_pmat.shape[2]
    te_mat = np.zeros(shape=(n_loci, n_loci))

    for loc1 in range(n_loci):
        YY = transition_pmat[loc1, loc1, :, :]
        for loc2 in range(n_loci):
            XY = transition_pmat[loc1, loc2, :, :]

            P = transition_pmat[loc1, loc2]
            p = 1
            p1 = np.linalg.matrix_power(P, p)
            p2 = np.linalg.matrix_power(P, p+1)
            # find the stationary distribution by increasing the power until no further change
            while not np.allclose(p1,p2) and not (p1==0).all() and not p > 100: 
                p += 1
                p1 = np.linalg.matrix_power(p1, p)
                p2 = np.linalg.matrix_power(p1, p+1)
            p1 = np.ones((n_tsteps))
            te_mat[loc1,loc2] = transfer_entropy(p1, XY, YY)

    return te_mat