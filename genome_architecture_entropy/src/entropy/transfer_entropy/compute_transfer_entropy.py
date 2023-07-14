import numpy as np
from scipy.spatial.distance import cdist


def transfer_entropy(N, k, pyYX, pyY):
    """Returns the transfer entropy of two probability distributions as defined by Schreiber et al. using KLD"""

    # where p(YY) is 0, p(XY) is implied to be 0 and the contribution of the log term is interpreted as 0 
    pq_ratio = np.divide(pyYX, pyY, out=np.zeros_like(pyYX), where=pyY!=0)

    # result is zero if pXY is 0 because lim_x->0+ xlog(x) = 0
    te = np.sum(1/(N-k) * np.log2(pq_ratio, out=np.zeros_like(pq_ratio), where=pq_ratio!=0))

    return te



def pw_transfer_entropy(seg_mat, hist_len=1, stattest=False):
    # max disrance of any two bins can not be larger than the largest number of occurences of a loci in all slices * number of bins
    n_tsteps, n_slice, n_loci = seg_mat.shape
    max_dist = np.max(np.sum(np.abs(seg_mat), axis=1)) * 2
    P = np.zeros(shape=(n_loci, n_loci, n_tsteps, n_tsteps))
    T = np.zeros(shape=(n_loci, n_loci))
    A = np.zeros(shape=(n_tsteps, n_tsteps, n_loci, n_loci))
    D = np.zeros(shape=(n_tsteps, n_tsteps))
    state1 = np.zeros(shape=(n_loci, n_slice))
    state2 = np.zeros(shape=(n_loci, n_slice))

    # compute all possible state combinations, k off the diagonal
    # remove upper and lower triangular repr. long time intervals with k+1 history length as diagonal cutoff
    NxN = np.ones(shape=(n_tsteps, n_tsteps))
    sparse_transitions = NxN - np.triu(NxN, +1 +hist_len) - np.tril(NxN, -1)
    transitions = np.transpose(np.nonzero(sparse_transitions)) 

    for t in range(len(transitions)):
        n1, n2 = transitions[t]
        for i in range(n_loci):
            for j in range(n_loci):
                state1[i] = seg_mat[n1, :, i]
                state2[j] = seg_mat[n2, :, j]

                if stattest:
                    vec = np.arange(0, n_loci)
                    np.random.shuffle(vec)
                    state2[j] = seg_mat[n2, :, vec[j]]
 
        # element-wise differences of all pairs (manhattan distance)
        #dist = state1[:, None] - state2[None, :]
        #S = np.sum(np.abs(dist), axis=-1)
        S = cdist(state1, state2, metric='cityblock') #faster than slicing

        # scale matrix from distance 0 to inf -> similarity 1 to 0 so that an all zero distance matrix becomes all ones
        A[n1, n2] = 1 - S / max_dist

    A = np.moveaxis(A, (2,3), (0,1))
    for i in range(n_loci):
        for j in range(n_loci):
            # P = D^-1 * A
            outdeg = np.sum(A[i, j], axis=1)[:, None] # weighted degrees as col vector
            np.fill_diagonal(D, outdeg) # degree matrix D
            P[i, j] = np.matmul(np.linalg.inv(D), A[i, j])
    
    for i in range(n_loci):
        for j in range(n_loci):
            C_xy = P[i, j]
            C_yy = P[i, i]
            T[i, j] = transfer_entropy(n_tsteps, hist_len, C_xy, C_yy)
    
    return T
