# %% Imports and helper function definitons
"""Imports and helper function definitons"""
import os
os.chdir('/Users/pita/Documents/Rene/GAM_project/genome_architecture_entropy/src/')
#os.chdir('/fast/AG_Metzger/rene/GAM_project/genome_architecture_entropy/src/')


import time
import numpy as np
import pandas as pd; pd.set_option("display.precision", 3)
import seaborn as sns
from matplotlib import pyplot as plt
import scipy
from scipy.stats import entropy, skew, kurtosis, spearmanr
from scipy.spatial.distance import hamming, pdist, squareform
from difflib import SequenceMatcher
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from nltk.util import ngrams
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score

from atpbar import atpbar
from mantichora import mantichora
import multiprocessing ;multiprocessing.set_start_method('fork', force=True)

import networkx as nx
import libpysal
from esda.moran import Moran

import entropy.transfer_entropy.compute_transfer_entropy as em


def patch_violinplot():
    from matplotlib.collections import PolyCollection
    ax = plt.gca()
    violins = [art for art in ax.get_children() if isinstance(art, PolyCollection)]
    for i in range(len(violins)):
        violins[i].set_linewidth(1.5)


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


def normalised_kendall_tau_distance(values1, values2):
    """Compute the Kendall tau distance."""
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1))

def spearman(seq):
    ordered_seq = np.sort(seq)
    rho, _ = spearmanr(seq, ordered_seq)
    return rho

def hamming_distance(seq):
    ordered_seq = np.sort(seq)
    return hamming(seq, ordered_seq)

def lcs_length(seq):
    ordered_seq = np.sort(seq)
    # Use difflib to get the LCS
    lcs = SequenceMatcher(None, seq, ordered_seq).find_longest_match(0, len(seq), 0, len(ordered_seq))
    return lcs.size

def jaccard_similarity(seq):
    ordered_seq = np.sort(seq)
    intersection = len(np.intersect1d(seq, ordered_seq))
    union = len(np.union1d(seq, ordered_seq))
    return intersection / union

def count_inversions(seq):
    return sum((seq[i] > seq[j]) for i in range(len(seq)) for j in range(i+1, len(seq)))

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


def compute_medoid(cluster_values):
    # Compute the pairwise distances between all sequences in the cluster
    distances = pdist(cluster_values, metric='hamming')
    # Convert the distances to a square form
    distances = squareform(distances)
    # Compute the sum of distances for each sequence
    total_distances = distances.sum(axis=1)
    # Find the sequence with the smallest total distance
    medoid = cluster_values[total_distances.argmin()]
    return medoid


# %%   1. Load model
'''1. Load model'''
path_model = '../data/toymodels/model10/'
model = 'toymodel10_multi_1000'
series = np.load(path_model + model + '.npy')

if len(series.shape) == 3:
    n_realisations = 1
    n_timesteps, n_loci, xy = series.shape

elif len(series.shape) == 4:
    n_realisations, n_timesteps, n_loci, xy = series.shape

print('>> loaded', model)
print('Realisations:', n_realisations, 
      '\nTimesteps:', n_timesteps, 
      '\nLoci:', n_loci)



# %%   2. Transform model
'''2. Transform model'''
if n_realisations > 1:
    # order='C' individual: 0..11 * 5
    # order='F') grouped 5*0, 5*1, 5*2, ..
    series_ = np.reshape(series, newshape=(n_realisations * n_timesteps, n_loci, 2), order='F') 
    print('>> transformed model')

n_cut_pre = 0
n_cut_post = 0
series_ = series_[n_realisations * n_cut_pre: n_timesteps * n_realisations - n_realisations * n_cut_post]
n_timesteps -= (n_cut_pre + n_cut_post)
print('>> removed', n_cut_pre + n_cut_post, 'timestep(s) for a total of', n_timesteps)



# %%    - Load segregation matrix
'''- Load segregation matrix'''
model_seg = '_seg_mats_1000_t%d' % (n_timesteps)
str_slice = ''#'2000/'
seg_mats = np.load(path_model + str_slice + model + model_seg + '.npy')

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
    series_ = series_[:-cut_diff]
    n_timesteps -= cut_diff
    print('>> removed', np.abs(cut_diff), 'timesteps from series')
elif cut_diff < 0:
    seg_mats = seg_mats[:cut_diff]
    print('>> removed', np.abs(cut_diff), 'timesteps from segregation matrix')

# set coords_model to the last timestep
coords_model = series_[-1].T



# %%    - Moran's I
'''Moran's I'''
#H = nx.DiGraph(TA)
#w_rook = libpysal.weights.lat2W(n_loci, n_loci, rook=True)
#w_queen = libpysal.weights.lat2W(n_loci, n_loci, rook=False)
#w = libpysal.weights.Rook.from_networkx(H)
#mi = Moran(TA, w_rook)
#print(mi.I, mi.p_sim)



# %%   6. Time order permutations
'''6. Time order permutations'''

# delete first timestep
n_timesteps -= 1
seg_mats = seg_mats[1:,:,:]

n_perm = np.math.factorial(n_timesteps)
print('Number of permutations = ', n_perm)
n_cores = 4
assert n_perm % n_cores == 0

vec = np.arange(0, n_perm)
np.random.shuffle(vec)
n_perm = 5040
#lexperms = faster_permutations(n_timesteps)[vec[:n_perm]] 

lexperms = lexographic_permutations(n_perm, np.arange(n_timesteps))
lexperms[0] = np.arange(n_timesteps)

def task(name, perms):
    te_symmetry = []
    permutations = []
    te_sums = []
    te_max = []
    te_min = []
    ta_sum = []
    kendall_corr = []
    te_std = []
    te_mean = []
    te_median = []
    te_var = []
    te_dist_entropy = []
    te_skewness = []
    te_kurtosis = []
    te_pca = []
    ta_mi = []
    te_mi = []
    ratio = []
    mean_row_entropy = []
    mean_col_entropy = []
    ta_var = []
    # Create StandardScaler object
    #scaler = StandardScaler()

    for i in atpbar(range(len(perms)), name=name):
        seq = perms[i]
        permutations.append(np.array(seq))
        kendall_corr.append(round(scipy.stats.kendalltau(np.arange(n_timesteps), seq)[0], 3))

        seg_mats_p = seg_mats[seq,:,:]
        te = -em.pw_transfer_entropy(seg_mats_p, 1).T

        te_sums.append(np.sum(te))
        #te_max.append(np.max(te))
        te_min.append(np.min(te[te > 0]))
        #te_std.append(np.std(te))
        te_var.append(np.var(te))
        #te_mean.append(np.mean(te))
        #te_median.append(np.median(te))
        te_symmetry.append(np.sum(np.triu(te) - np.tril(te)))
        #te_skewness.append(skew(te.flatten()))
        te_kurtosis.append(kurtosis(te.flatten()))
        mean_row_entropy.append(np.mean([entropy(row) for row in te]))
        #mean_col_entropy.append(np.mean([entropy(col) for col in te.T]))
        #te_mi.append(Moran(te, w_rook).I)

        #scaled_te_matrix = scaler.fit_transform(te) # Fit and transform the data to center it and scale to unit variance
        #pca = PCA(n_components=1) # Create PCA object
        #pca.fit(scaled_te_matrix) # Fit the scaled data
        #te_pca.append(pca.explained_variance_ratio_[0]) # Extract the explained variance ratio of the first component

        te_values = te.flatten()
        hist = np.histogram(te_values, bins='auto', density=True)
        te_dist_entropy.append(entropy(hist[0]))

        ta = 0.5 * (te - te.T)
        tap = ta[ta > 0]
        ta_sum.append(tap.sum())
        #ratio.append(np.sum(te) / np.sum(tap))
        #ta_mi.append(Moran(ta, w_rook).I)
        #ta_var.append(np.var(ta))

    d = {'Permutation': permutations,
        'TE symmetry': te_symmetry,
        'Variance TE': te_var,
        'Entropy TE': te_dist_entropy,
        #'Skewness TE': te_skewness,
        'Kurtosis TE': te_kurtosis,
        #'PCA TE': te_pca,
        #'Median TE': te_median,
        #'Mean TE': te_mean,
        #'Standard deviation': te_std,
        'Total sum TE': te_sums,
        'Total asymmetrical TE sum': ta_sum,
        'Ratio': ratio,
        #'Max TE': te_max,
        'Min TE': te_min,
        #'Moran\'s I TE': te_mi,
        #'Moran\'s I TA': ta_mi,
        'Mean row entropy': mean_row_entropy,
        #'Mean column entropy': mean_col_entropy,
        #'Variance TA': ta_var,
        'Kendall': kendall_corr,}

    return (d)


with mantichora(nworkers=n_cores) as mcore:
    start = time.time()
    for i in range(n_cores):
        mcore.run(task, 'task %d' % i, lexperms[int(n_perm / n_cores) * i: int(n_perm / n_cores) * (i+1)])
    returns = mcore.returns()
end = time.time()
print('Execution time:', (end - start) / 60, 'minutes')
print('Execution time per permutation: %d seconds' % float((end - start) / n_perm))

# join dictionaries from list of dictionaries
d = {}
for t in range(n_cores):
    for key, value in returns[t].items():
        d.setdefault(key, []).extend(value)
d.pop('Ratio', None)
df_perm = pd.DataFrame(d)


name_df = 'df_perm_%d.csv' % n_perm
df_perm.to_csv(path_model + model + model_seg + name_df)
#df = df.round(pd.Series(6, df.drop('Permutation', axis=1).columns))

# transform the last 5 columns to z scores
#from scipy.stats import zscore
#df_perm.iloc[:, -6:] = df_perm.iloc[:, -6:].apply(zscore)

# scale last 10 columns between 0 and 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_perm.iloc[:, 1:-1] = scaler.fit_transform(df_perm.iloc[:, 1:-1])

# Add permutation distance measures
# convert number in [] in permutation to int


df_perm['Permutation'] = df_perm['Permutation'].apply(np.array)


df_perm['Spearman'] = df_perm['Permutation'].apply(spearman)


df_perm['Kendall_abs'] = df_perm['Kendall'].apply(abs)
df_perm['Hamming'] = df_perm['Permutation'].apply(hamming_distance)
df_perm['LCS'] = df_perm['Permutation'].apply(lcs_length)
df_perm['Inversions'] = df_perm['Permutation'].apply(count_inversions)




# %%    - Load permutation results
'''- Load permutation results'''
df_perm = pd.read_csv(path_model + model + model_seg + 'df_perm_5040.csv', index_col=0)



# %%    - Correlate all columns (Pearson's r)
'''- Correlate all columns (Pearson's r)'''
val = []
var1 = []
var2 = []
for col1 in df_perm.iloc[:, 1:]:
    pos = df_perm.columns.get_loc(col1)
    for col2 in df_perm.iloc[:, pos:]:
        corr = df_perm[col1].corr(df_perm[col2])
        if col1!=col2 and np.abs(corr) > 0.3:
            val.append(round(corr, 2))
            var1.append(col1)
            var2.append(col2)

d2 = {'Pearsons r': val,
      'Variable 1': var1,
      'Variable 2': var2}

df_corr = pd.DataFrame(d2).sort_values('Pearsons r', key=abs, ascending=False)

# correlate 'Hamming', 'Inversions, 'LCS' vs all other variables with pearson's r
table = df_perm.corr(method='pearson').sort_values('Hamming', key=abs, ascending=False).loc[['Hamming', 'Inversions', 'LCS', 'Spearman', 'Kendall', 'cluster'], :]
pd.set_option('display.float_format', lambda x: '%.2f' % x)
table.loc['abs sum'] = table.abs().sum(axis=0)

df_corr.groupby(['Variable 1', 'Variable 2']).value_counts()

df = df_perm.sort_values('Inversions', key=abs)
df = df.drop(['Mean column entropy'], axis=1)
df["One shuffle away"] = df['Inversions'] < 2
df["Backwards"] = df['Inversions'] > 19



# %%    - Linear regression
'''- Linear regression'''
method = 'Inversions'
#X = df_perm[['Standard deviation', 'Total sum TE', 'Total asymmetrical TE sum ']]
X = df_perm.drop(['Permutation', 'Hamming', 'LCS', 'Inversions', 'Spearman', 'Kendall'], axis=1)
y = df_perm[method]

m = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scores = cross_val_score(m, X_train, y_train, scoring='neg_mean_squared_error', cv=4)

reg = m.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print('The target is:', method)
for i,j in enumerate(reg.coef_):
    print(f"Feature {X_train.columns[i]}: Score: {j}")

print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
print('Variance of target:', np.round(np.var(y), 3))

# Evaluate on the test set
test_predictions = reg.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
print(f"The Mean Squared Error on the Test set is: {test_mse:.3f}")
print(f"Cross-validation Mean Squared Error on Training set: {-scores.mean():.3f} (+/- {scores.std():.3f})")




# %%    - Violin plots
'''- Violin plots'''
plt.rcParams["figure.figsize"] = (12,6)
plt.rcParams['figure.dpi'] = 75
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set_style('whitegrid')
cmap = sns.color_palette(["#FFBC3F", "#EE8051", "#D875B5", "#A266EC", "#5F95F7"])
cmap = sns.color_palette(["#FFBC3F", "#EE8051", "#D3D3D3", "#A266EC", "#5F95F7"]) # grey for insufficient

for y_var in vars:
    sns.violinplot(data=df, x="Kendall correlation vs ordered sequence", y=y_var, inner="box", area="count", cut=1, palette=cmap)
    sns.stripplot(x=df["Kendall correlation vs ordered sequence"][df["One shuffle away"]], y=df[y_var][df["One shuffle away"]], 
                color='white', edgecolor='#5F95F7', zorder=2, linewidth=1, size=5, jitter=False)
    sns.despine(left=True, bottom=True)

    patch_violinplot()
    file_violin = path_model + model + model_seg + '_violin_' + y_var + '.png'
    plt.savefig(file_violin)

    plt.show()
    plt.close()



# %%    - Scatter plots
'''- Scatter plots'''
plt.rcParams["figure.figsize"] = (10,10)
plt.rcParams['figure.dpi'] = 100
sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2.5})
sns.set_style('whitegrid')

cmap_ibm = sns.color_palette(["#648FFF", "#785EF0", "#E23689", "#F5711E", "#FFB000"])
cmap = sns.color_palette(["#FFBC3F", "#EE8051", "#D875B5", "#A266EC", "#5F95F7"])
cmap = sns.color_palette(["#FFBC3F", "#EE8051", "#D3D3D3", "#A266EC", "#5F95F7"]) # grey for insufficient
extend = 0.05

for y_var1 in vars:
    pos = vars.index(y_var1)
    for y_var2 in vars[:pos]:
        #g = sns.JointGrid(data=df, x=y_var1, y=y_var2, space=0, hue='Spearman cat', palette=cmap)
        #g.plot_joint(sns.scatterplot, s=30, alpha=0.5, linewidth=0, edgecolor='black', hue='Spearman cat', palette=cmap, legend=False)
        #g.plot_marginals(sns.kdeplot, fill=True, alpha=0.4, linewidth=0, edgecolor=None, palette=cmap)
        #g.refline(x=df[y_var1].quantile(0.1), y=df[y_var2].quantile(0.1), color='black', linestyle='--', linewidth=.5)
        #g.ax_joint.scatter(x=df[df["One shuffle away"]][y_var1], y=df[df["One shuffle away"]][y_var2], s=20, alpha=0.6, linewidth=2, edgecolor='#5F95F7', color="white", zorder=1)
        #g.ax_joint.scatter(x=filtered_df2[y_var1], y=filtered_df2[y_var2], s=75, alpha=0.4, linewidth=0.8, edgecolor='black', color="white", zorder=1)
        #legend_labels = [t.get_text() for t in g.ax_joint.get_legend().get_texts()] labels=[*legend_labels, '99th percentile']
        #g.ax_joint.legend(title='Kendall Correlation', loc='lower left', bbox_to_anchor=(1, 1), ncol=1, frameon=True)

        g = sns.scatterplot(data=df[df['classes'].isin(['.'])], x=y_var1, y=y_var2, s=50, alpha=0.5, linewidth=0, hue='classes', palette=cmap, legend=False)
        g = sns.scatterplot(data=df[df['classes'].isin(['--'])], x=y_var1, y=y_var2, s=50, alpha=0.5, linewidth=0, hue='classes', palette=cmap, legend=False)
        g = sns.scatterplot(data=df[df['classes'].isin(['-'])], x=y_var1, y=y_var2, s=50, alpha=0.5, linewidth=0, hue='classes', palette=cmap, legend=False)
        g = sns.scatterplot(data=df[df['classes'].isin(['++'])], x=y_var1, y=y_var2, s=50, alpha=0.5, linewidth=0, hue='classes', palette=cmap, legend=False)
        g = sns.scatterplot(data=df[df['classes'].isin(['+++'])], x=y_var1, y=y_var2, s=50, alpha=0.5, linewidth=0, hue='classes', palette=cmap, legend=False)
        g = sns.scatterplot(x=df[df["One shuffle away"]][y_var1], y=df[df["One shuffle away"]][y_var2], s=50, alpha=1, linewidth=2, edgecolor='#5F95F7', color="white", zorder=1)

        # extend xlim and ylim
        #xlim = g.ax_joint.get_xlim()
        #xrange = xlim[1] - xlim[0]
        #ylim = g.ax_joint.get_ylim()
        #yrange = ylim[1] - ylim[0]
        #g.ax_joint.set_xlim(xlim[0] + extend*xrange, xlim[1] - extend*xrange)
        #g.ax_joint.set_ylim(ylim[0] + extend*yrange, ylim[1] - extend*yrange)

        # rename axes
        #g.ax_joint.set_xlabel(y_var1)
        #g.ax_joint.set_ylabel(y_var2)

        file_scatter = path_model + model + model_seg + '_scatter_' + y_var1 + '_' + y_var2 + '.png'
        #plt.savefig(file_scatter)

        plt.show()
        plt.close()



# %%    - Bivariate KDE plots
'''- Bivariate KDE plots'''
plt.rcParams["figure.figsize"] = (8,8)
plt.rcParams['figure.dpi'] = 100
sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2.5})
sns.set_style('dark')

alpha = 0.6
treshold = 0.1
levels = 20
extend = 0.05

for y_var1 in vars:
    pos = vars.index(y_var1)
    for y_var2 in vars[:pos]:
        g = sns.kdeplot(data=df[df['classes'].isin(['.'])], x=y_var1, y=y_var2, color="#D3D3D3", fill=True, thresh=treshold, levels=levels, cbar=False, cbar_kws={"orientation": "horizontal", "shrink": .5, "label": "Count"})
        g = sns.kdeplot(data=df[df['classes'].isin(['--'])], x=y_var1, y=y_var2, color="#EE8051", antialiased=True, alpha=alpha, fill=True, thresh=treshold, levels=levels, cbar=False, cbar_kws={"orientation": "horizontal", "shrink": .5, "label": "Count"})
        g = sns.kdeplot(data=df[df['classes'].isin(['---'])], x=y_var1, y=y_var2, color="#FFBC3F", antialiased=True, alpha=alpha, fill=True, thresh=treshold, levels=levels, cbar=False, cbar_kws={"orientation": "horizontal", "shrink": .5, "label": "Count"})
        g = sns.kdeplot(data=df[df['classes'].isin(['++'])], x=y_var1, y=y_var2, color="#A266EC", antialiased=True, alpha=alpha, fill=True, thresh=treshold, levels=levels, cbar=False, cbar_kws={"orientation": "horizontal", "shrink": .5, "label": "Count"})
        g = sns.kdeplot(data=df[df['classes'].isin(['+++'])], x=y_var1, y=y_var2, color="#5F95F7", antialiased=True, alpha=alpha, fill=True, thresh=treshold, levels=levels, cbar=False, cbar_kws={"orientation": "horizontal", "shrink": .5, "label": "Count"})
        g.scatter(x=df[df["One shuffle away"]][y_var1], y=df[df["One shuffle away"]][y_var2], alpha=1, s=20, linewidth=2, color='white', edgecolor="#5F95F7", zorder=10)
        #g.legend(title='Kendall Correlation', loc='lower left', bbox_to_anchor=(1, 1), ncol=1, frameon=True)

        # extend xlim and ylim
        xlim = g.get_xlim()
        xrange = xlim[1] - xlim[0]
        ylim = g.get_ylim()
        yrange = ylim[1] - ylim[0]
        g.set_xlim(xlim[0] + extend*xrange, xlim[1] - extend*xrange)
        g.set_ylim(ylim[0] + extend*yrange, ylim[1] - extend*yrange)

        for line in g.get_lines():
            line.set_alpha(0)

        file_kde = path_model + model + model_seg + '_kde_' + y_var1 + '_' + y_var2 + '.png'
        plt.savefig(file_kde)

        plt.show()
        plt.close()



# %%    - Metrics cluster analysis
'''- Metrics cluster analysis'''''
# Parameters
n_clusters = 6

# Bin the quality metrics
metrics = list(df.columns[1:9])
df_bin = pd.DataFrame()
df_bin['Permutation'] = df['Permutation']

for metric in metrics:
    df_bin[f'metric_{metric}_bin'] = pd.cut(df[f'{metric}'], bins=10, labels=False)

# Group by the binned quality metrics
grouped = df_bin.groupby([f'metric_{metric}_bin' for metric in metrics])

# Perform clustering
cluster = KMeans(n_clusters)
df['cluster'] = cluster.fit_predict(df_bin[df_bin.columns[1:-1]]) + 1
df_perm['cluster'] = df['cluster']



# %%   - Metrics cluster analysis
'''- Metrics cluster analysis'''''

# filter out he Permutations in 'Backwards' and 'One shuffle away' and their ratio between variance and total sum
df_filter = df[df["Backwards"]]
df_filter = df_filter.append(df[df["One shuffle away"]])
#df_filter['ratio'] = df_filter['Variance TE'] / df_filter['Total sum TE']
#drop all columns except for 'Permutation', 'cluster' and 'ratio'
df_filter = df_filter[['Permutation', 'cluster', 'Variance TE', 'Total sum TE']]
df_filter

# summary statistics for each cluster
df_describe = df_perm.groupby('cluster').describe()

# CI between clusters for each metric mean
df_ci = pd.DataFrame()
for metric in metrics:
    df_ci[f'{metric}_ci'] = df_describe[metric]['mean'] + 1.96 * df_describe[metric]['std'] / np.sqrt(df_describe[metric]['count'])
df_ci = df_ci.T
df_ci

# Krustal Wallis test for each metric
import scipy.stats as stats
results = pd.DataFrame(columns=['Measure', 'H statistic', 'p-value'])

# Loop over each measure
for measure in ['TE symmetry', 'Variance TE', 'Total sum TE', 'Kurtosis TE', 'Total asymmetrical TE sum', 'Min TE', 'Mean row entropy', 'Mean column entropy', 'Entropy TE']:
    grouped = df_perm.groupby('cluster')[measure]
    
    # Create a list of lists, where each sublist contains the measure values for a single cluster
    clusters = [data.tolist() for _, data in grouped]
    
    # Perform the Kruskal-Wallis H test
    H, p_val = stats.kruskal(*clusters)
    
    # Store the results in the results DataFrame
    results = results.append({'Measure': measure, 'H statistic': H, 'p-value': p_val}, ignore_index=True)

# Display the results
print(results)




# %%   - SSD
'''- SSD'''
sse = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(df_bin[df_bin.columns[1:-1]])
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel('Number of clusters k')
plt.ylabel('Sum of squared distance')
plt.show()



# %%   - Silhouette index
'''- Silhouette index'''
sil = []
list_k = list(range(2, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(df_bin[df_bin.columns[1:-1]])
    preds = km.predict(df_bin[df_bin.columns[1:-1]])
    
    sil.append(silhouette_score(df_bin[df_bin.columns[1:-1]], preds))

# Plot sil against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sil, '-o')
plt.xlabel('Number of clusters k')
plt.ylabel('Silhouette Score')
plt.show()



# %%   - Plot clusters
'''- Plot clusters'''
sns.set_theme(style='darkgrid')
sns.set_context('paper', font_scale=2)

plt.rcParams["figure.figsize"] = (10,10)
plt.rcParams['figure.dpi'] = 150
sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2.5})

alpha = 0.6
treshold = 0.05
levels = 20
lgd = False

my_palette_dark = ['#45A7B0', '#9A3C4D', '#654D9A', '#B47141', '#348632', '#31619B']
my_palette_light = ['#78C7CE', '#D9A3B4', '#B4A3D9', '#DEBDA3', '#A3DEA6', '#96A9D9']
my_palette_medium = ['#5EB2BB', '#C26B7E', '#8B6BC2', '#D19B5A', '#5EB25E', '#5E8BB2']

col_to_remove = ['Permutation', 'Hamming', 'LCS', 'Inversions', 'Spearman', 'Kendall', 'cluster', 'Min TE', 'Mean row entropy', 'Mean column entropy']
vars = [col for col in df_perm.columns.tolist() if col not in col_to_remove]
vars = ['Variance TE', 'Entropy TE']

for y_var1 in vars:
    pos = vars.index(y_var1)
    for y_var2 in vars[:pos]:
        g = sns.kdeplot(data=df, x=y_var1, y=y_var2, hue='cluster', palette=my_palette_dark, antialiased=True, alpha=alpha, fill=True, thresh=treshold, levels=levels, cbar=False, cbar_kws={"orientation": "horizontal", "shrink": .5, "label": "Count"}, legend=False)
        #g = sns.scatterplot(data=df, x=y_var1, y=y_var2, hue='cluster', palette=my_palette, zorder=0)
        g = sns.scatterplot(data=df, x=df[df["One shuffle away"]][y_var1], y=df[df["One shuffle away"]][y_var2], alpha=1, edgecolor='white', linewidth=1, hue='cluster', style='One shuffle away', palette=my_palette_medium, zorder=1, legend=lgd, markers=['X', 'o'], s=100)
        g = sns.scatterplot(data=df, x=df[df["Backwards"]][y_var1], y=df[df["Backwards"]][y_var2], alpha=1, edgecolor='white', linewidth=1, hue='cluster', style='Backwards', zorder=1, palette=my_palette_medium, legend=False, markers=['o', 'X'], s=100)
        #plt.legend(bbox_to_anchor=(1, 0), loc='lower left', ncol=1)
        g.grid(True, which='both', color='white')
        plt.grid()
        plt.show()
        plt.close()



# %%   - Per cluster sequences
medoids = []
mode = []
flip_flag = []
cluster_id = []
ratio = []
flip = [True, False]
for flip in flip:
    for c in range(1, n_clusters+1):
        clust = df[df['cluster'] == c][['Permutation', 'Spearman']]
        # add 1 to the permutation to avoid 0
        #clust['Permutation'] = clust['Permutation'] + 1
        # Transform the sequences into a 2-dimensional array
        cluster_values = np.stack(clust['Permutation'].values)
        direction = clust['Spearman'].values
        # count number of positives and negative
        n_pos = np.sum(direction > 0)
        n_neg = np.sum(direction < 0)
        ratio.append(n_pos / n_neg)
        # flip the sequences to the majority direction
        if flip:
            if n_pos > n_neg:
                cluster_values[direction < 0] = np.flip(cluster_values[direction < 0], axis=1)
            else:
                cluster_values[direction > 0] = np.flip(cluster_values[direction > 0], axis=1)
        
        medoids.append(compute_medoid(cluster_values))
        mode.append(scipy.stats.mode(cluster_values, axis=0, keepdims=True)[0][0])
        flip_flag.append(flip)
        cluster_id.append(c)

# make pandas dataframe
d3 = {'cluster': cluster_id, 'medoid': medoids, 'mode': mode, 'ratio': ratio ,'flip': flip_flag}
df_cluster_seq = pd.DataFrame(data=d3)
df_cluster_seq



# %%   - Plot sequences
plt.rcParams["figure.figsize"] = (14,2)
plt.rcParams['figure.dpi'] = 300
sns.set_theme(style='darkgrid')
sns.set_context('paper', font_scale=1.3)

#subplot with shared axis
fig, axes = plt.subplots(nrows=1, ncols=6, sharex=True, sharey=True)
axes = axes.flatten()

for i, (index, row) in enumerate(df_cluster_seq.iterrows()):
    if row['flip']:
        axes[i].plot(row['medoid'], label=row['cluster'], color=my_palette_medium[row['cluster']-1], linewidth=3)
        axes[i].scatter(range(len(row['medoid'])), row['medoid'], label=row['cluster']-1, color=my_palette_dark[row['cluster']-1], zorder=2, s=40)
        
ticks = range(0,7)
labels = [str(i+1) for i in ticks]

for ax in axes:
    #ax.set_facecolor('#ededed')
    #ax.grid(color='white')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(' ' * len(ticks))
    ax.set_aspect('equal')

    #if ax == axes[4] or ax == axes[5]:
    #    ax.set_xlabel('Time', fontsize=16)
    #if ax == axes[2]:
    #    ax.set_ylabel('Timepoint in sequence', labelpad=10, fontsize=16)
    

plt.subplots_adjust(hspace=0.1, wspace=0.05)

plt.show()

# %%    - Subsequence cluster analysis
'''- Subsequence cluster analysis'''
# Parameters
n_clusters_sub = 5
n_clusters = 4
ngram_length = 4

# Function to extract n-grams with positions
def extract_ngrams(sequence, n):
    return [(i, gram) for i, gram in enumerate(ngrams(sequence, n))]

# Extract n-grams from sequences
ngrams_list = []
for index, row in df.iterrows():
    sequence = row['Permutation']
    for position, ngram in extract_ngrams(sequence, ngram_length):
        ngrams_list.append([index, position] + list(ngram))

# Create DataFrame of n-grams
df_ngrams = pd.DataFrame(ngrams_list, columns=['index', 'position', 'ngram1', 'ngram2', 'ngram3', 'ngram4'])

# Prepare for clustering: convert ngrams to strings and label encode them
df_ngrams['ngram_str'] = df_ngrams[['ngram1', 'ngram2', 'ngram3', 'ngram4']].apply(lambda row: ''.join(row.values.astype(str)), axis=1)
le = LabelEncoder()
df_ngrams['ngram_encoded'] = le.fit_transform(df_ngrams['ngram_str'])

# Perform clustering
kmeans = KMeans(n_clusters_sub)
df_ngrams['cluster'] = kmeans.fit_predict(df_ngrams[['position', 'ngram_encoded']])

# Create a pivot table counting the number of n-grams in each cluster for each sequence
pivot = pd.pivot_table(df_ngrams, values='ngram_encoded', index='index', columns='cluster', aggfunc='count', fill_value=0)

# Normalize the counts by the total number of n-grams in each sequence
pivot_normalized = pivot.div(pivot.sum(axis=1), axis=0)

# Perform clustering on the sequences
kmeans_sequences = KMeans(n_clusters)
df['cluster'] = kmeans_sequences.fit_predict(pivot_normalized)

# Create the scatterplot
y_var1 = 'Entropy TE'
y_var2 = 'Mean row entropy'
g = sns.scatterplot(data=df, x=y_var1, y=y_var2, hue='cluster', palette='tab10')
g = sns.scatterplot(x=df[df["One shuffle away"]][y_var1], y=df[df["One shuffle away"]][y_var2], alpha=1, linewidth=2, edgecolor='#5F95F7', color="white", zorder=1)
for cluster in range(n_clusters):
    print(cluster, scipy.stats.mode(np.stack(df[df['cluster'] == cluster]['Permutation'], axis=0), axis=0)[0])


