# %% '''Import and helper function definitons'''
import os
#os.chdir('/Users/pita/Documents/Rene/GAM_project/genome_architecture_entropy/src/')
os.chdir('/fast/AG_Metzger/rene/GAM_project/genome_architecture_entropy/src/')

import time
import scipy
import numpy as np
import pandas as pd; pd.set_option("display.precision", 2)
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
#import libpysal
#from esda.moran import Moran
from atpbar import atpbar, register_reporter, find_reporter, flush
from mantichora import mantichora
import multiprocessing ;multiprocessing.set_start_method('fork', force=True)

import entropy.transfer_entropy.compute_transfer_entropy as em
import toymodel.sampling
import entropy.transfer_entropy.transition as tb
import entropy.transfer_entropy.plotting as plotting


def lexographic_permutations(length, sequence):
    perms = np.empty((length, len(sequence)), dtype=np.uint8)
    l = 0
    while l < length:
        diff_seq_idx = np.where(np.diff(sequence) > 0)[0]
        if not len(diff_seq_idx) == 0:
            i = diff_seq_idx[-1]
            j = np.where(sequence > sequence[i])[0][-1]
            # swap the values at i and j
            sequence[i], sequence[j] = sequence[j], sequence[i]
            # reverse the sequence after i
            sequence[i + 1:] = sequence[i + 1:][::-1]
            perms[l] = sequence.copy()
            l += 1
        else:
            sequence = sequence[::-1]

    return perms


def faster_permutations(n):
    # empty() is fast because it does not initialize the values of the array
    # order='F' uses Fortran ordering, which makes accessing elements in the same column fast
    perms = np.empty((np.math.factorial(n), n), dtype=np.uint8, order='F')
    perms[0, 0] = 0

    rows_to_copy = 1
    for i in range(1, n):
        perms[:rows_to_copy, i] = i
        for j in range(1, i + 1):
            start_row = rows_to_copy * j
            end_row = rows_to_copy * (j + 1)
            splitter = i - j
            perms[start_row: end_row, splitter] = i
            perms[start_row: end_row, :splitter] = perms[:rows_to_copy, :splitter]  # left side
            perms[start_row: end_row, splitter + 1:i + 1] = perms[:rows_to_copy, splitter:i]  # right side

        rows_to_copy *= i + 1

    return perms


def deviations(sequence):
    deviations = np.ediff1d(sequence.astype(int))
    # 0,1,2,3,4,5 = 5

    # 5,4,3,0,1,2 = -3

    # 5,4,3,2,1,0 = -5

    # 5,0,4,1,3,2 = -3

    # 0,5,4,3,2,1 = 1 => -2
    #  5 -1-2-2-2 = -2
    # higher score to multiple in line

    direction = deviations.sum()
    # sum of deviations of deviations 
    sdd = np.abs(np.ediff1d(deviations).astype(int)).sum()
    return sdd, direction


def normalised_kendall_tau_distance(values1, values2):
    """Compute the Kendall tau distance."""
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    
    return ndisordered / (n * (n - 1))


def vn_eig_entropy(rho):
    from scipy import linalg as la
    EV = la.eigvals(rho)

    # Drop zero eigenvalues so that log2 is defined
    my_list = [x for x in EV.tolist() if x]
    EV = np.array(my_list)

    log2_EV = np.matrix(np.log2(EV))
    EV = np.matrix(EV)
    S = -np.dot(EV, log2_EV.H)
    
    return(S)


# %% '''Load model'''
path_model = '../data/toymodels/model4/'
model = 'toymodel4_2_multi500'
series = np.load(path_model + model + '.npy')

if len(series.shape) == 3:
    n_realisations = 1
    n_timesteps = series.shape[0]
    n_loci = series.shape[1]

elif len(series.shape) == 4:
    n_realisations = series.shape[0]
    n_timesteps = series.shape[1]
    n_loci = series.shape[2]

print('>> loaded', model)
print('Realisations:', n_realisations, 
      '\nTimesteps:', n_timesteps, 
      '\nLoci:', n_loci)



# %% '''Quick coordinate viewer'''
plt.rcParams["figure.figsize"] = (6,3)
plt.rcParams['figure.dpi'] = 80

for realisation in range(5):
    for state in range(series.shape[1]):
        plt.plot(series[realisation].T[0,:,state], series[realisation].T[1,:,state], '-k')
    plt.show()
    plt.clf()



# %% '''Transform model'''
if n_realisations > 1:
    # order='C' individual: 0..11 * 5
    # order='F') grouped 5*0, 5*1, 5*2, ..
    series = np.reshape(series, newshape=(n_realisations * n_timesteps, n_loci, 2), order='F') 
    print('>> transformed model')

n_cut = 4
series = series[n_realisations * n_cut:]# -n_realisations * n_cut]
n_timesteps -= (n_cut)
print('>> removed', n_cut, 'timesteps for a total of', n_timesteps)



# %% '''Compute segregation matrix'''
n_states = n_realisations * n_timesteps
n_slice = 4
wdf_lb = 0.1
seg_mats = np.empty((n_states, n_slice, series.shape[1]))

for state in tqdm(range(n_states)):
    coords_model = series[state].T
    seg_mat = toymodel.sampling.slice(coords_model, n_slice, wdf_lb)
    seg_mats[state] = seg_mat

print('>> Segregation matrix computed with', n_slice, 'slices and', wdf_lb, 'lower bound window detection frequency')

if n_realisations > 1:
    seg_mats = np.reshape(seg_mats, newshape=(n_timesteps, n_realisations * n_slice, n_loci), order='F') # only for grouped!
    print('>> Realisations have been grouped for', n_realisations * n_slice, 'for each timestep')

np.save(path_model + model + '_seg_mats_%d_t%d.npy' % (n_realisations * n_slice, n_timesteps), seg_mats)



# %% '''Load segregation matrix'''
model_seg = '_seg_mats_2000_t7.npy'
seg_mats = np.load(path_model + model + model_seg)

# Assert that the segregation matrix is correct
if n_realisations > 1:
    n_slice = int(seg_mats.shape[1] / n_realisations)
else: 
    n_slice = seg_mats.shape[1]
# assert than n_slice is an integer
assert n_slice == int(n_slice), "n_slice is not an integer"
assert seg_mats.shape[2] == n_loci, 'Loci do not match'

print('>> loaded', model_seg)
print('Slices:', n_slice)

cut_diff = n_timesteps - seg_mats.shape[0]
if int(cut_diff) != 0: print('>> series n_cut is not matching segregation matrix n_cut. Difference:', cut_diff)
# match n_cut of segmats to n_cut of series if necessary
if cut_diff > 0:
    series = series[:-cut_diff]
    n_timesteps -= cut_diff
    print('>> removed', np.abs(cut_diff), 'timesteps from series')
elif cut_diff < 0:
    seg_mats = seg_mats[:cut_diff]
    print('>> removed', np.abs(cut_diff), 'timesteps from segregation matrix')

# set coords_model to the last timestep
coords_model = series[-1].T



# %% '''Window detection statistics'''
n_slice_ = n_realisations * n_slice
detect_sum = [2,3,4,5,6,7,8,9,10]
n_detect = np.zeros(shape=(n_timesteps, len(detect_sum)))
n_unique = []
wdf_mean = []
wdf_min = []
wdf_max = []

print('Timesteps:', n_timesteps, '\nRealisations:', n_realisations, '\nSlices:', n_slice, '\nLoci:', n_loci)
for t in range(n_timesteps):
    wdf = np.sum(seg_mats[t], axis=0) / n_slice_
    n_unique.append(100 * np.unique(seg_mats[t], return_counts=True, axis=0)[1].shape[0] / n_slice_)

    for cnt, val in enumerate(detect_sum):
        n_detect[t, cnt] = np.sum(np.sum(seg_mats[t], axis=1) == val) / n_slice_

    wdf_mean.append(np.mean(100*wdf))
    wdf_min.append(np.min(100*wdf))
    wdf_max.append(np.max(100*wdf))

d = {'% Mean wdf': wdf_mean, 
    'Min wdf': wdf_min,
    'Max wdf': wdf_max,
    'Unique': n_unique}

idx = np.arange(n_timesteps+n_cut)[n_cut:].tolist()
df = pd.DataFrame(d, index=idx)
df2 = pd.DataFrame(n_detect, columns=detect_sum, index=idx)
df_ = pd.concat([df, df2], axis=1)
df_



# %% '''Transfer Entropy calculation and plotting'''

seg_mats_ = seg_mats#[np.array(seq),:,:]; print('Sequence = ', seq)
hist_len = 7

# backwards time order
#seg_mats_ = np.flip(seg_mats_)

# random time order
#seg_mats_ = np.random.RandomState().permutation(seg_mats)

# randomize loci 
#seg_mats_ = np.moveaxis(seg_mats_, 2, 0)
#seg_mats_ = np.random.RandomState().permutation(seg_mats_)
#seg_mats_ = np.moveaxis(seg_mats_, 0, 2)

tprobs = tb.bin_probs(seg_mats_, n_bin=n_loci, hist_len=hist_len)
te_mat = em.multivariate_transfer_entropy(tprobs)
te_mat_asym = te_mat - te_mat.T
#te_mask = np.ma.masked_less(te_mat_asym_0, 0)

print('History length = ', hist_len)
print('TE sum =', np.sum(te_mat))
print('TE direction =', np.sum(np.triu(te_mat) - np.tril(te_mat)))
print('TE asymmetry =', np.sum(te_mat_asym[te_mat_asym > 0]))
print('TE median =', np.median(te_mat))
print('TE mean =', np.mean(te_mat))
print('TE std =', np.std(te_mat))
plotting.heatmaps(te_mat, te_mat_asym)



# %% '''Effective TE'''
hist_len = 7
temp = np.empty(shape=(100, n_loci, n_loci))
# calculating the small samplre contribution
for itr in tqdm(range(100)):
    seg_mats_ = np.moveaxis(seg_mats, 2, 0)
    seg_mats_p = np.random.RandomState().permutation(seg_mats_)
    seg_mats_p = np.moveaxis(seg_mats_p, 0, 2)
    probs = tb.bin_probs(seg_mats_p, n_bin=n_loci, hist_len=hist_len)
    temp[itr] = em.multivariate_transfer_entropy_(probs)
te_mat_perm = np.mean(temp, axis=0)

print('History length = ', hist_len)
print('SSC TE sum =', np.sum(te_mat_perm))
print('SSC TE direction =', np.sum(np.triu(te_mat_perm) - np.tril(te_mat_perm)))
print('SSC TE median =', np.median(te_mat_perm))
print('SSC TE mean =', np.mean(te_mat_perm))
print('SSC TE std =', np.std(te_mat_perm))

te_eff = te_mat - te_mat_perm
print('Eff. TE = TE - TE(SSC) = ', np.sum(te_eff)) 
plotting.heatmaps(te_mat_perm, te_eff)

#te_eff = te_eff + np.abs(te_eff.min())
#np.fill_diagonal(te_eff, 0)



# %% '''Moran's I'''
#H = nx.DiGraph(te_mat_asym)
#w_rook = libpysal.weights.lat2W(n_loci, n_loci, rook=True)
#w_queen = libpysal.weights.lat2W(n_loci, n_loci, rook=False)
##w = libpysal.weights.Rook.from_networkx(H)
#mi = Moran(te_mat_asym, w_rook)
#print(mi.I, mi.p_sim)



# %% '''Time permuation'''
n_perm = 5040
hist_len = 7
sequences = []
te_sums = []
te_asym_sums = []
cross_corr = []
kendall_corr = []
kendall_dist = []
te_net = []
te_std = []
te_mean = []
te_median = []
mi_r = []
mi_r_asym = []
seq = np.arange(n_timesteps)
seq_ = seq.copy()

#all_permutations = faster_permutations(n_timesteps) # 12 max
#idx = np.random.RandomState().permutation(np.arange(all_permutations.shape[0])).tolist()
lexperms = lexographic_permutations(n_perm, np.arange(n_timesteps))

for i in tqdm(range(n_perm)):
    if i == 0:
        seq = np.arange(n_timesteps)
        seq_ = seq.copy()

    sequences.append(seq)
    #cross_corr.append(np.correlate(seq_, seq)[0])
    #kendall_dist.append(normalised_kendall_tau_distance(seq_, seq))
    kendall_corr.append(round(scipy.stats.kendalltau(seq_, seq)[0], 3))

    seg_mats_p = seg_mats[seq,:,:]
    tprobs = tb.bin_probs(seg_mats_p, n_bin=n_loci, hist_len=hist_len)
    te = em.multivariate_transfer_entropy(tprobs)

    te_sums.append(np.sum(te))
    te_std.append(np.std(te))
    te_mean.append(np.mean(te))
    te_median.append(np.median(te))
    te_net.append(np.sum(np.triu(te) - np.tril(te)))

    te_asym = te - te.T
    #mean = np.mean(te_asym[te_asym > 0])
    te_asym_sums.append(te_asym[te_asym > 0].sum())

    #mi_r.append(Moran(te, w_rook).I)
    #mi_r_asym.append(Moran(te_asym, w_rook).I)

    seq = lexperms[i]
    #seq = np.random.RandomState().permutation(seq)

d = {'Permutation': sequences,
     #'Seq. Kendall dist.': kendall_dist/max(kendall_dist),
     #'Seq. cross corr.': cross_corr/max(cross_corr),
     'Kendalls tau-b (perm.)': kendall_corr,
     'TE direction': te_net,
     'TE median': te_median,
     'TE mean': te_mean,
     'TE std': te_std,
     'TE sum': te_sums,
     'Asym TE sum': te_asym_sums,
     #'Morans I (rook)': mi_r,
     #'Morans I asym (rook)': mi_r_asym,
     }
df_perm = pd.DataFrame(d)
#df = df.round(pd.Series(6, df.drop('Permutation', axis=1).columns))
df_perm = df_perm.sort_values('Kendalls tau-b (perm.)', key=abs)
np.save('/fast/AG_Metzger/rene/GAM_project/genome_architecture_entropy/data/results/model4_2_4x500_df_perm_5040.npy', df_perm)

name_df = 'df_perm_%d.csv' % n_perm
df_perm.to_csv(path_model + model + model_seg + name_df)


# %% Parallel process time order permutations
n_perm = 5040
n_cores = 16
hist_len = 7
lexperms = lexographic_permutations(n_perm, np.arange(n_timesteps))

def task(name, perms):
    te_net = []
    permutations = []
    te_sums = []
    te_asym_sums = []
    kendall_corr = []
    te_std = []
    te_mean = []
    te_median = []

    for i in atpbar(range(len(perms)), name=name):
        seq = perms[i]
        permutations.append(seq)
        kendall_corr.append(round(scipy.stats.kendalltau(np.arange(n_timesteps), seq)[0], 3))

        seg_mats_p = seg_mats[seq,:,:]
        tprobs = tb.bin_probs(seg_mats_p, n_bin=n_loci, hist_len=hist_len)
        te = em.multivariate_transfer_entropy(tprobs)

        te_sums.append(np.sum(te))
        te_std.append(np.std(te))
        te_mean.append(np.mean(te))
        te_median.append(np.median(te))
        te_net.append(np.sum(np.triu(te) - np.tril(te)))

        te_asym = te - te.T
        te_asym_sums.append(te_asym[te_asym > 0].sum())

    d = {'Permutation': permutations,
         'Kendalls tau-b (perm.)': kendall_corr,
         'TE direction': te_net,
         'TE median': te_median,
         'TE mean': te_mean,
         'TE std': te_std,
         'TE sum': te_sums,
         'Asym TE sum': te_asym_sums,}

    return (d)


with mantichora(nworkers=n_cores) as mcore:
    start = time.time()
    for i in range(n_cores):
        mcore.run(task, 'task %d' % i, lexperms[int(n_perm / n_cores) * i: int(n_perm / n_cores) * (i+1)])
    returns = mcore.returns()
end = time.time()
print('Execution time: %d seconds' % (end - start))


# join dictionaries from list of dictionaries
d = {}
for t in range(n_cores):
    for key, value in returns[t].items():
        d.setdefault(key, []).extend(value)

df_perm = pd.DataFrame(d)
df_perm = df_perm.sort_values('Kendalls tau-b (perm.)', key=abs)

name_df = 'df_perm_%d.csv' % n_perm
df_perm.to_csv(path_model + model + model_seg + name_df)
#df = df.round(pd.Series(6, df.drop('Permutation', axis=1).columns))



# %% '''Correlate all columns (Pearson's r)'''
val = []
var1 = []
var2 = []

df = df_perm

for col1 in df.iloc[:, 1:]:
    pos = df.columns.get_loc(col1)
    for col2 in df.iloc[:, pos:]:
        corr = df[col1].corr(df[col2])
        if corr < 0.99 and np.abs(corr) > 0.1:
            val.append(round(corr, 2))
            var1.append(col1)
            var2.append(col2)

d2 = {'Pearsons r': val,
      'Variable 1': var1,
      'Variable 2': var2}

df_corr = pd.DataFrame(d2).sort_values('Pearsons r', key=abs, ascending=False)
print(df_corr)

# group observatios by Kendall corr. into equally sized bins
n_bin = 5
df["Kendall correlation to ordered sequence"] = pd.qcut(df['Kendalls tau-b (perm.)'], n_bin, 
labels=["strong negative", "weak negative", "none", "weak positive", "strong positive"],)
#labels=["- -", "-", "none", "+", "+ +"])

# group observations by by Kendall corr. into fixed bins
df["Kendall correlation vs ordered sequence"] = pd.cut(df['Kendalls tau-b (perm.)'],[-1, -0.75, -0.5, 0.5, 0.75, 1], #[0.7, 0.75, 0.8, 0.85, 0.95, 1,], #
#labels=['0.70 - 0.75', '0.75 - 0.80', '0.80 - 0.85', '0.85 - 0.95', '0.95 - 1.00'])
labels=["-1 to -0.75", "-0.75 to -0.5", "-0.5 to 0.5", "0.5 to 0.75", "0.75 to 1"],)

df["99th percentile"] = df['Kendalls tau-b (perm.)'] > df['Kendalls tau-b (perm.)'].quantile(0.995)
df['TE sum zscore'] = (df['TE sum'] - df['TE sum'].mean()) / df['TE sum'].std(ddof=0)

#filtered_df = df[(df['TE std'] < df['TE std'].quantile(0.2)) & 
#                 (df['TE direction'] < df['TE direction'].quantile(0.025)) & 
#                 (df['TE median'] > df['TE median'].quantile(0.5))]


#filtered_df2 = df[(df['TE std'] < df['TE std'].quantile(0.8)) & 
#                  (df['TE direction'] > df['TE direction'].quantile(0.85)) &
#                  (df['TE sum zscore'] < 0.5) & (df['TE sum zscore'] > 0)]
#majority_sequence = scipy.stats.mode(np.stack(filtered_df2['Permutation'], axis=0), axis=0)
#print(majority_sequence)



# %% '''Plot violin plots'''
def patch_violinplot():
    from matplotlib.collections import PolyCollection
    ax = plt.gca()
    violins = [art for art in ax.get_children() if isinstance(art, PolyCollection)]
    for i in range(len(violins)):
        violins[i].set_linewidth(1.5)

plt.rcParams["figure.figsize"] = (12,6)
plt.rcParams['figure.dpi'] = 75
sns.set_context("notebook", font_scale=1.3, rc={"lines.linewidth": 2.5})
sns.set_style('whitegrid')

cmap = sns.color_palette(["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"])
cmap = sns.color_palette(["#FFBC3F", "#EE8051", "#D875B5", "#A266EC", "#5F95F7"])
vars = ['TE direction', 'TE sum', 'TE std']

for y_var in vars:
    sns.violinplot(data=df, x="Kendall correlation vs ordered sequence", y=y_var, inner="box", area="count", cut=1, palette=cmap)
    sns.stripplot(x=df["Kendall correlation vs ordered sequence"][df["99th percentile"]], y=df[y_var][df["99th percentile"]], 
                color='#5F95F7', zorder=2, linewidth=2, size=10, jitter=False)
    sns.despine(left=True, bottom=True)

    patch_violinplot()
    plt.show()



# %% '''Plot scatter plots'''
plt.rcParams["figure.figsize"] = (12,12)
plt.rcParams['figure.dpi'] = 100
sns.set_context("notebook", font_scale=1.3, rc={"lines.linewidth": 2.5})
sns.set_style('white')

cmap_ibm = sns.color_palette(["#648FFF", "#785EF0", "#E23689", "#F5711E", "#FFB000"])
cmap = sns.color_palette(["#FFBC3F", "#EE8051", "#D875B5", "#A266EC", "#5F95F7"])
cmap = sns.color_palette(["#FFBC3F", "#EE8051", "#D3D3D3", "#A266EC", "#5F95F7"])
vars = ['TE direction', 'TE median', 'TE mean']
extend = 0.1
filter = df[df['Kendall correlation vs ordered sequence'].isin(['strong positive', 'weak positive'])]

for y_var1 in vars:
    pos = vars.index(y_var1)
    for y_var2 in vars[:pos]:
        g = sns.JointGrid(data=df, x=y_var1, y=y_var2, space=0, hue='Kendall correlation vs ordered sequence', palette=cmap)
        g.plot_joint(sns.scatterplot, s=30, alpha=1, linewidth=0, edgecolor='black', hue='Kendall correlation vs ordered sequence', palette=cmap)
        g.plot_marginals(sns.kdeplot, fill=True, alpha=0.4, linewidth=0, edgecolor=None, palette=cmap)
        #g.refline(x=df[y_var1].quantile(0.1), y=df[y_var2].quantile(0.1), color='black', linestyle='--', linewidth=.5)
        g.ax_joint.scatter(x=df[df["99th percentile"]][y_var1], y=df[df["99th percentile"]][y_var2], s=30, alpha=1, linewidth=0.5, edgecolor='black', color="none", zorder=1)
        #g.ax_joint.scatter(x=filtered_df2[y_var1], y=filtered_df2[y_var2], s=75, alpha=0.4, linewidth=0.8, edgecolor='black', color="white", zorder=1)
        #legend_labels = [t.get_text() for t in g.ax_joint.get_legend().get_texts()] labels=[*legend_labels, '99th percentile']
        g.ax_joint.legend(title='Seq. Kendall Correlation', loc='lower left', bbox_to_anchor=(1, 1), ncol=1, frameon=True)

        # extend xlim and ylim
        xlim = g.ax_joint.get_xlim()
        xrange = xlim[1] - xlim[0]
        ylim = g.ax_joint.get_ylim()
        yrange = ylim[1] - ylim[0]
        g.ax_joint.set_xlim(xlim[0] + extend*xrange, xlim[1] - extend*xrange)
        g.ax_joint.set_ylim(ylim[0] + extend*yrange, ylim[1] - extend*yrange)

        plt.show()

# %% '''Bivariate KDE plots'''
plt.rcParams["figure.figsize"] = (12,12)
plt.rcParams['figure.dpi'] = 100
sns.set_context("notebook", font_scale=1.3, rc={"lines.linewidth": 2.5})
sns.set_style('dark')

alpha = 0.6
treshold = 0.2
levels = 20

for y_var1 in vars:
    pos = vars.index(y_var1)
    for y_var2 in vars[:pos]:
        g = sns.kdeplot(data=df[df['Kendall correlation vs ordered sequence'].isin(['-0.5 to 0.5'])], x=y_var1, y=y_var2, color="#D3D3D3", fill=True, thresh=treshold, levels=levels, cbar=False, cbar_kws={"orientation": "horizontal", "shrink": .5, "label": "Count"})
        g = sns.kdeplot(data=df[df['Kendall correlation vs ordered sequence'].isin(['-0.75 to -0.5'])], x=y_var1, y=y_var2, color="#EE8051", antialiased=True, alpha=alpha, fill=True, thresh=treshold, levels=levels, cbar=False, cbar_kws={"orientation": "horizontal", "shrink": .5, "label": "Count"})
        g = sns.kdeplot(data=df[df['Kendall correlation vs ordered sequence'].isin(['-1 to -0.75'])], x=y_var1, y=y_var2, color="#FFBC3F", antialiased=True, alpha=alpha, fill=True, thresh=treshold, levels=levels, cbar=False, cbar_kws={"orientation": "horizontal", "shrink": .5, "label": "Count"})
        g = sns.kdeplot(data=df[df['Kendall correlation vs ordered sequence'].isin(['0.5 to 0.75'])], x=y_var1, y=y_var2, color="#A266EC", antialiased=True, alpha=alpha, fill=True, thresh=treshold, levels=levels, cbar=False, cbar_kws={"orientation": "horizontal", "shrink": .5, "label": "Count"})
        g = sns.kdeplot(data=df[df['Kendall correlation vs ordered sequence'].isin(['0.75 to 1'])], x=y_var1, y=y_var2, color="#5F95F7", antialiased=True, alpha=alpha, fill=True, thresh=treshold, levels=levels, cbar=False, cbar_kws={"orientation": "horizontal", "shrink": .5, "label": "Count"})
        g.scatter(x=df[df["99th percentile"]][y_var1], y=df[df["99th percentile"]][y_var2], alpha=0.8, s=30, linewidth=0.5, color='white', edgecolor="#5F95F7", zorder=10)
        #g.legend(title='Seq. Kendall Correlation', loc='lower left', bbox_to_anchor=(1, 1), ncol=1, frameon=True)

        # extend xlim and ylim
        xlim = g.get_xlim()
        xrange = xlim[1] - xlim[0]
        ylim = g.get_ylim()
        yrange = ylim[1] - ylim[0]
        g.set_xlim(xlim[0] + extend*xrange, xlim[1] - extend*xrange)
        g.set_ylim(ylim[0] + extend*yrange, ylim[1] - extend*yrange)

        for line in g.get_lines():
            line.set_alpha(0)

        plt.show()

# %% '''Significance testing'''
# test paramater
reps = 40
cores = 4

# te parameter
nbin = n_loci
history_l = n_timesteps

reporter = find_reporter()
ran_test_te = np.zeros(shape=(reps, te_mat.shape[0], te_mat.shape[1]))
ran_test_te_asym = np.zeros(shape=(reps, te_mat.shape[0], te_mat.shape[1]))
bool_arr = np.zeros_like(ran_test_te)

for rep in atpbar(range(0, reps, cores), name = multiprocessing.current_process().name):
#for rep in tqdm(range(0, reps, cores)):
    #sequence = np.random.RandomState().permutation(sequence)
    
    with multiprocessing.Pool(cores, register_reporter, [reporter]) as pool:
        workloads = []
        for job in range(cores):
            seg_mats_ = np.moveaxis(seg_mats, 2, 0)
            seg_mats_p = np.random.RandomState().permutation(seg_mats_)
            seg_mats_p = np.moveaxis(seg_mats_p, 0, 2)
            workload = [(seg_mats_p, nbin, history_l)] 
            workloads.extend(workload)
        res_probs = pool.starmap(tb.bin_probs, workloads)
        flush()

    with multiprocessing.Pool(cores, register_reporter, [reporter]) as pool:
        workloads = []
        for job in range(cores):
            workloads.append(res_probs[job])
        res_te = pool.map(em.multivariate_transfer_entropy, workloads)
        flush()

    #for job, array in enumerate(res_probs):
    #    ran_test_te[rep+job, :, :] = em.all_transfer_entropy(array)

    for item, array in enumerate(res_te):
        ran_test_te[rep+item, :, :] = array


for sample in range(reps):
    #ran_test_te_asym[sample] = ran_test_te[sample] - ran_test_te[sample].T
    bool_arr[sample] = np.greater_equal(np.abs(ran_test_te[sample]), np.abs(te_mat))

props = bool_arr.sum(axis=0) / reps
plotting.heatmaps(props - np.diag(np.diag(props)))

mean = np.mean(np.triu(props, k=1))
median = np.median(props)

print('mean proportion =', mean)
print('median proportion =', median)
print('max proportion =', np.max(np.triu(props, k=1)))

mask = props < 0.05

te_asym = np.subtract(te_mat, te_mat.T, where=mask, out=np.zeros_like(te_mat))
plotting.heatmaps(te_mat, te_asym)



