# %%
'Imports'
import numpy as np
import entropy_measures as em
import toymodel.trans_probs as tb
#from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from atpbar import atpbar, register_reporter, find_reporter, flush

import multiprocessing
multiprocessing.set_start_method('fork', force=True)


def create_cont_series(length, param, xself_dep, yran):
    '''
    length - series length
    param - number of variables per process
    xself_dep - self-dependency of x
    yran - randomness on y
    '''

    X = np.zeros(shape=(length+1, param))
    Y = np.zeros(shape=(length+1, param))

    for i in range(1, length+1):
        k_base = np.random.normal(loc=0, scale=2)
        for k in range(0, param):
            X[i, k] = np.around(xself_dep * (X[i-1, k] + k_base) + (1 - xself_dep) * np.random.normal(loc=0, scale=2), 5)
            Y[i, k] = np.around((1-yran) * X[i-1, k] + yran * np.random.normal(loc=0, scale=2), 5)

    return X,Y


def create_binary_series(length, param, xself_dep, yran):
    X = np.zeros(shape=(length+1, param))
    Y = np.zeros(shape=(length+1, param))

    for i in range(1, length+1):
        k_base = np.random.choice([0,1], p=[0.5, 0.5])
        for k in range(0, param):
            choice_x = [X[i-1, k], k_base, 0, 1]
            choice_y = [X[i-1, k], 0, 1]
            X[i, k] = np.random.choice(choice_x, p=[xself_dep, (1-xself_dep)/2, (1-xself_dep)/4, (1-xself_dep)/4])
            Y[i, k] = np.random.choice(choice_y, p=[1-yran, yran/2, yran/2])

    return X,Y


def test(X, Y, hist_len):
    global probs
    global te_mat
    length = X.shape[0]
    param = X.shape[1]
    nbin = nbin = 2*param

    process = np.dstack((X,Y)).reshape(length, 1, 2*param, order='F')
    sequence = np.linspace(0, length-1, length)
    probs = tb.bin_probs(process, sequence, nbin, hist_len)
    te_mat = em.all_transfer_entropy(probs)
    te_yx = np.sum(np.triu(te_mat))
    te_xy = np.sum(np.tril(te_mat))
    te_net_xy = te_xy - te_yx

    return te_net_xy


def permutation_test(length, param, xself_dep, rep, hist_len, nperm=100):
    alpha = 0.05 # significance treshold
    perm_res = np.zeros(shape=(nperm , 1))
    res = np.zeros(shape=(rep, 2))

    #for j in atpbar(range(rep), name = multiprocessing.current_process().name):
    for j in range(rep):
        X, Y = create_cont_series(length, param, xself_dep, yran)

        '''
        if (j == 0) and param < 4:
            plt.plot(np.linspace(0, length, length+1), X, label = "X")
            plt.plot(np.linspace(0, length, length+1), Y, label = "Y", linestyle='dotted')
            plt.legend()
            plt.show()
        '''
        res[j, 0] = test(X, Y, hist_len)
        # X is permuted nperm times and the significance result saved to each series realization te entry
        for perm in range(nperm):
            X_perm = np.random.permutation(X)
            perm_res[perm] = test(X_perm, Y, hist_len)

        # two sided p-test: surrogate TE larger/smaller than original in percent
        proportion_myte_left = (perm_res >= res[j,1]).sum() / rep
        proportion_myte_right = (perm_res <= res[j,1]).sum() / rep
        res[j, 1] = int(proportion_myte_left < alpha or proportion_myte_right < alpha)

    return res


def grid_search(rep, nperm):
    reporter = find_reporter()
    grid = np.zeros(shape=(4,4,4,4))

    for l, length in enumerate([5, 10, 25, 50]):
        for p, param in enumerate([1, 2, 5, 10]):
            print('\nlength =', length, ' ', l+1, '/4')
            print('param  =', param, ' ', p+1, '/4')
            for k, hist_len in enumerate([1, 2, 5, length]):

                with multiprocessing.Pool(4, register_reporter, [reporter]) as pool:
                    workloads = [(length, param, 0, rep, hist_len, nperm),
                                (length, param, 0.2, rep, hist_len, nperm),
                                (length, param, 0.5, rep, hist_len, nperm),
                                (length, param, 0.9, rep, hist_len, nperm)]
                    res = pool.starmap(permutation_test, workloads)
                    flush()
                for item, array in enumerate(res):
                    grid[l, p, k, item] = np.sum(array, axis=0)[1] / rep
                #for s, xself_dep in enumerate([0, 0.2, 0.5, 0.8]):
                    #res = permutation_test(length, param, xself_dep, rep, hist_len, nperm)
                    #grid[l, p, k, s] = np.sum(res, axis=0)[1] / rep

    return grid


# %% 
'Compute TE of lin. dependent process X -> Y'

length = 50
param = 4
xself_dep = 0.3
global yran; yran = 0
#global hist_len; hist_len = 1

rep = 10 # number of iterations
nperm = 100 # number of permutations for signficance testing

#res = permutation_test(length, param, xself_dep, rep, hist_len, nperm)
#print('mean TE =', np.around(np.mean(res, axis=0)[0], 4))
#print('std =', np.around(np.std(res, axis=0)[0], 4))
#print('signif. ratio =', np.sum(res, axis=0)[1] / rep)
#print('\nlength =', length, '; n parameter =', param, '; history length = ', hist_len, ';\nX self-depency =', xself_dep, '; Y randomness =', yran)

grid = grid_search(rep, nperm)