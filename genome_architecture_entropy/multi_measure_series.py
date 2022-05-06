# %% import
'''Segregation Analysis'''
import os
from re import A
import numpy as np
import pathlib
import seaborn as sns
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt, colors
from tqdm.auto import tqdm

import cosegregation_internal as ci
import entropy_measures as em
import toymodel.conf_space as cs
import toymodel.trans_probs as tb

os.chdir('/Users/pita/Documents/Rene/GAM_project/genome_architecture_entropy/')
path_series_out = str(pathlib.Path.cwd() / 'toymodel/out/')

def plot_large_ensemble(coords_model, matrices_vmax_dict, mi_sums_vmax, state_H, *args):
    '''4x2 plot fxn'''
    # Generate a mask for the upper triangle from value shape, k=1 to see diagonal line in heatmap
    any_mat = matrices_vmax_dict['Norm. Mutual Information (NMI)'][0]
    mask_shape = (any_mat.shape[1], any_mat.shape[1])
    mask = np.triu(np.ones(mask_shape, dtype=bool), k=0)

    # style and colors
    sns.set_theme(style='white')
    scatter_cmap = sns.color_palette('dark:dodgerblue', as_cmap=True)
    cmap_legend_1 = plt.cm.ScalarMappable(cmap=sns.color_palette('dark:dodgerblue', as_cmap=True))
    #cmap_legend_2 = plt.cm.ScalarMappable(cmap=sns.color_palette('flare', as_cmap=True))
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=mi_sums_vmax[1])

    # dynamic coordinate boundaries
    xmin = np.min(coords_model[0])
    xmax = np.max(coords_model[0])
    ymin = np.min(coords_model[1])
    ymax = np.max(coords_model[1])
    height = ymax - ymin
    width = xmax - xmin
    length = coords_model.shape[1]
    scaling = length / width # for scatter point size

    # set subtitle text
    text_info = ''
    for i in state_H:
        text_info = text_info + i + str(state_H[i]) + ']'

    # set figure
    fig = plt.figure(figsize=(16, 8))
    plt.figtext(x=0.35, y=0.05, s=text_info, fontsize=16, weight='bold', )
    ax1 = plt.subplot(2,4,(1,2))
    ax2 = plt.subplot(2,4,3)
    ax3 = plt.subplot(2,4,4)
    ax4 = plt.subplot(2,4,5)
    ax5 = plt.subplot(2,4,6)
    ax6 = plt.subplot(2,4,7)
    ax7 = plt.subplot(2,4,8)
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    
    # set coordinate plot
    axes[0].plot(coords_model[0], coords_model[1], '-k') 
    axes[0].scatter(coords_model[0], coords_model[1], s=50*1.8*scaling, c=mi_sums_vmax[0], norm=normalize, cmap=scatter_cmap)
    axes[0].set_aspect('equal', 'datalim')
    axes[0].tick_params(labelleft=False, labelbottom=False)
    axes[0].set_title('Chromatin model colored by accumulated MI', loc='left', fontsize=14, y=0.94)
    axes[0].axis('off')
    plt.colorbar(cmap_legend_1, ax=axes[0], shrink=0.5).outline.set_visible(False)
    #plt.colorbar(cmap_legend_2, ax=[axes[2], axes[6]]).outline.set_visible(False)

    # set the heatmap plots
    for index, (key, value) in enumerate(matrices_vmax_dict.items()):
        axes[index+1].set_title(key, loc='left', fontsize=14)
        axes[index+1].tick_params(labelleft=False, labelbottom=False)
        if key == 'Univariate Transfer Entropy':
            set_mask = None
            set_vmin = None
            set_cmap = sns.color_palette("mako_r", as_cmap=True)
            #sns.color_palette("vlag", as_cmap=True)
        else: 
            set_mask = mask
            set_vmax = value[1]
            set_vmin = 0
            set_cmap = sns.color_palette('flare', as_cmap=True)

        sns.heatmap(value[0],
                    ax=axes[index+1],
                    mask=set_mask,
                    cmap=set_cmap,
                    robust=True,
                    square=True,
                    linewidths=0,
                    cbar_kws={"shrink": .5},
                    vmax=set_vmax,
                    vmin=set_vmin)

    plt.tight_layout(pad=4, h_pad=2, w_pad=1)
    #plt.savefig(path_series_out + args[0], dpi=100)
    #plt.close(fig)

def scale(a, vmax):
    a = a - a.min()
    a = a - np.diag(np.diag(a))
    # divide calculation anywhere 'where' a does not equal zero
    a = np.divide(a, vmax, out=np.zeros_like(a), where = vmax != 0)

    return a


# %% 
'''Interpolate Series'''
fps = 30 
series = np.load('toymodel/md_soft/out/toymodel.npy')
series_extended = np.empty((series.shape[0], fps, series.shape[1], series.shape[2]))


for state in range(series.shape[0] - 1):
    states = np.array([series[state], series[state + 1]])
    realizations = np.empty((fps, states.shape[1], states.shape[2]))

    for t in range(0, fps):
        scale = t / (fps - 1)
        weight = np.full_like(states[0], scale)
        weights = np.array([1 - weight, weight])
        realizations[t] = np.average(states, axis=0, weights=weights)
    
    series_extended[state,:,:,:] = realizations

series_extended = series_extended.reshape(-1, series.shape[1], 2)
np.save('toymodel/md_soft/out/toymodel_interpolated.npy', series_extended)



'''Series Calculations'''
n_slice = 20000
series = np.load('toymodel/md_soft/out/toymodel.npy')
series_extended = np.load('toymodel/md_soft/out/toymodel_interpolated.npy')
vmax = [0, 0, 0, 0, 0]
seg_mats = np.empty((series.shape[0], n_slice, series.shape[1]))

# precompute max values over series
for state in tqdm(range(series.shape[0])):
    coords_model = series[state].T
    seg_mat = cs.slice(coords_model, n_slice)

    je_mat = em.all_joint_entropy(seg_mat.T)
    mi_mat = em.mutual_information(je_mat)
    mi_mat = mi_mat - np.diag(np.diag(mi_mat))
    vmax[0] = max(mi_mat.max(), vmax[0])

    mi_sums = np.sum(mi_mat, axis=0)
    vmax[2] = max(mi_sums.max(), vmax[2])

    coseg_mat = ci.dprime_2d(seg_mat.T.astype(int), seg_mat.T.astype(int))
    coseg_mat = coseg_mat - np.diag(np.diag(coseg_mat))
    vmax[1] = max(coseg_mat.max(), vmax[1])
    
    pmi_mat = em.npmi_2d_fast(seg_mat.T.astype(int), seg_mat.T.astype(int))
    pmi_mat = np.nan_to_num(pmi_mat, copy=False, nan=0.0)
    pmi_mat = pmi_mat - np.diag(np.diag(pmi_mat))
    vmax[3] = max(pmi_mat.max(), vmax[3])

    seg_mats[state] = seg_mat
    sequence = np.linspace(0, state, state+1) # order of realizations is ordered
    probs = tb.bin_probs(seg_mats, sequence, nbin=36)
    te_mat = em.all_transfer_entropy(probs)
    vmax[4] = max(te_mat.max(), vmax[4])

# compute all information measures:
# shannon-, joint- entropy, mutual information, linkage disequilibiurm and pointwise mi
te_mat_last = np.empty_like(te_mat)
for state in tqdm(range(series.shape[0])):
    coords_model = series[state].T
    seg_mat = cs.slice(coords_model, n_slice)

    # measure calculation
    entropy = str(np.around(em.shannon_entropy(seg_mat), 2))

    je_mat = em.all_joint_entropy(seg_mat.T)
    mi_mat = em.mutual_information(je_mat)
    mi_mat = mi_mat - np.diag(np.diag(mi_mat))

    mi_sums = np.sum(mi_mat, axis=0)

    coseg_mat = ci.dprime_2d(seg_mat.T.astype(int), seg_mat.T.astype(int))
    coseg_mat = coseg_mat.clip(min=0)

    pmi_mat = em.npmi_2d_fast(seg_mat.T.astype(int), seg_mat.T.astype(int))
    pmi_mat = np.nan_to_num(pmi_mat, copy=False, nan=0.0)

    if state % 30 == 0:
        seg_mats[state // 30] = seg_mat
        sequence = np.linspace(0, state // 30, state // 30 +1) # order of realizations is ordered
        probs = tb.bin_probs(seg_mats, sequence, nbin=36)
        te_mat = em.all_transfer_entropy(probs)
        te_mat_last = te_mat
    else:
        #te_mat = te_mat_last for extended series, delete below
        seg_mats[state] = seg_mat
        sequence = np.linspace(0, state, state +1) # order of realizations is ordered
        probs = tb.bin_probs(seg_mats, sequence, nbin=36)
        te_mat = em.all_transfer_entropy(probs)
        te_mat_last = te_mat

    # normalize matrices 
    nmi_mat = scale(mi_mat, vmax[0])
    ncoseg_mat = scale(coseg_mat, vmax[1])
    npmi_mat = scale(pmi_mat, vmax[3])
    te_mat = scale(te_mat, vmax[4])
    te_mat[te_mat == 0] = np.nan

    #compute diff matrices
    ndiff_mi_ld_mat = ncoseg_mat - nmi_mat
    ndiff_mat_pmi_mat = npmi_mat - nmi_mat

    # prepare plot function call
    matrices_vmax_dict = {'Norm. Linkage Disequilibirum (NLD)': [ncoseg_mat, 1],
                          'Norm. Pointwise MI (PMI)': [npmi_mat, 1],
                          'Multivariate Transfer Entropy': [te_mat, 1],
                          'Norm. Mutual Information (NMI)': [nmi_mat, 1],
                          'Difference: Δ(NMI, NLD)': [ndiff_mi_ld_mat, 1],
                          'Difference Δ(NMI, NPMI)': [ndiff_mat_pmi_mat, 1]}
    mi_sums_vmax = [mi_sums, vmax[2]]
    state_H = {'Timestep [T = ': state // 30, '; Global entropy [H = ': entropy}
    file = '/state%03d.png' % (state)

    plot_large_ensemble(coords_model, matrices_vmax_dict, mi_sums_vmax, state_H, file)


# %% 
'''Tad boundary detection'''
# compute row wise sums
mi_sums = np.sum(mi_mat, axis=0)

coseg_mat = ci.dprime_2d(seg_mat.T.astype(int), seg_mat.T.astype(int))
coseg_mat = coseg_mat - np.diag(np.diag(coseg_mat))
coseg_sums = np.sum(coseg_mat, axis=0)

pmi_mat = pmi_mat - np.diag(np.diag(pmi_mat))
pmi_sums = np.sum(pmi_mat, axis=0)

# figure making
# Generate a mask for the upper triangle from value shape, k=1 to see diagonal line in heatmap
mask_shape = (nmi_mat.shape[1], nmi_mat.shape[1])
mask = np.triu(np.ones(mask_shape, dtype=bool), k=0)

# style and colors
sns.set_theme(style='white')
scatter_cmap = sns.color_palette('dark:dodgerblue', as_cmap=True)
cmap_legend_1 = plt.cm.ScalarMappable(cmap=sns.color_palette('dark:dodgerblue', as_cmap=True))
normalize_mi = matplotlib.colors.Normalize(vmin=mi_sums.min(), vmax=mi_sums.max())
normalize_coseg = matplotlib.colors.Normalize(vmin=coseg_sums.min(), vmax=coseg_sums.max())
normalize_npmi = matplotlib.colors.Normalize(vmin=pmi_sums.min(), vmax=pmi_sums.max())

# set figure
fig = plt.figure(figsize=(16, 8))
#plt.figtext(x=0.35, y=0.05, s=text_info, fontsize=16, weight='bold', )
ax1 = plt.subplot(2,3,1)
ax2 = plt.subplot(2,3,2)
ax3 = plt.subplot(2,3,3)
ax4 = plt.subplot(2,3,4)
ax5 = plt.subplot(2,3,5)
ax6 = plt.subplot(2,3,6)
ax7 = fig.add_axes([0.92, 0.55, 0.006, 0.3])
axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

# set coordinate plot
axes[0].plot(coords_model[0], coords_model[1], '-k') 
axes[0].scatter(coords_model[0], coords_model[1], s=100, c=mi_sums, norm=normalize_mi, cmap=scatter_cmap)
axes[0].set_title('Chromatin model colored by NMI sum', loc='center', fontsize=14, y=0.94)
axes[0].set_aspect('equal', 'datalim')
axes[0].tick_params(labelleft=False, labelbottom=False)
axes[0].axis('off')

axes[1].plot(coords_model[0], coords_model[1], '-k') 
axes[1].scatter(coords_model[0], coords_model[1], s=100, c=coseg_sums, norm=normalize_coseg, cmap=scatter_cmap)
axes[1].set_title('Chromatin model colored by NLD sum', loc='center', fontsize=14, y=0.94)
axes[1].set_aspect('equal', 'datalim')
axes[1].tick_params(labelleft=False, labelbottom=False)
axes[1].axis('off')

axes[2].plot(coords_model[0], coords_model[1], '-k') 
axes[2].scatter(coords_model[0], coords_model[1], s=100, c=pmi_sums, norm=normalize_npmi, cmap=scatter_cmap)
axes[2].set_title('Chromatin model colored by NPMI sum', loc='center', fontsize=14, y=0.94)
axes[2].set_aspect('equal', 'datalim')
axes[2].tick_params(labelleft=False, labelbottom=False)
axes[2].axis('off')

plt.colorbar(cmap_legend_1, cax=axes[6], shrink=0.82).outline.set_visible(False)

# set 3 heatmap plots
axes[3].set_title('Mutual Information (MI)', loc='left', fontsize=14)
axes[3].tick_params(labelleft=False, labelbottom=False)
sns.heatmap(mi_mat,
            ax=axes[3],
            mask=mask,
            cmap=sns.color_palette('flare', as_cmap=True),
            robust=True,
            square=True,
            linewidths=0,
            cbar_kws={"shrink": .82},
            vmin=0)

axes[4].set_title('Norm. Linkage Disequilibirum (NLD)', loc='left', fontsize=14)
axes[4].tick_params(labelleft=False, labelbottom=False)
sns.heatmap(coseg_mat,
            ax=axes[4],
            mask=mask,
            cmap=sns.color_palette('flare', as_cmap=True),
            robust=True,
            square=True,
            linewidths=0,
            cbar_kws={"shrink": .82},
            vmin=0)

axes[5].set_title('Norm. Pointwise MI (NPMI)', loc='left', fontsize=14)
axes[5].tick_params(labelleft=False, labelbottom=False)
sns.heatmap(pmi_mat,
            ax=axes[5],
            mask=mask,
            cmap=sns.color_palette('flare', as_cmap=True),
            robust=True,
            square=True,
            linewidths=0,
            cbar_kws={"shrink": .82},
            vmin=0)

# %%
'''Transfer entropy network '''
te_mat_asym = te_mat - te_mat.T

sns.heatmap(te_mat_asym,
            cmap=sns.color_palette("Blues", as_cmap=True),
            robust=True,
            square=True,
            linewidths=0,
            cbar_kws={"shrink": .82})
plt.show()

te_mat_asym[te_mat_asym < 0.06] = 0

def nmax_cube(array, n):
    row_idx = np.argpartition(array, range(n))[:, :-n-1: -1]
    col_idx = np.argpartition(array.T, range(n))[:, :-n-1: -1]
    array_rmins = array
    array_cmins = array
    np.put_along_axis(array_rmins, row_idx, 0, axis=1)
    np.put_along_axis(array_rmins, col_idx, 0, axis=2)
    array_nmax = array - array_rmins - array_cmins

    return array_nmax

H = nx.DiGraph(te_mat_asym)
pos = coords_model.T
edges, weights = zip(*nx.get_edge_attributes(H,'weight').items())
vmax_w = np.nanmax(list(weights))
weights_norm = weights / vmax_w
weights_list = list(np.nan_to_num(weights_norm, copy=False, nan=0.0))
edges_list = list(edges)
nx.draw(H, pos=pos, node_color='b', edge_color=weights_list, width=0.5, edge_cmap=plt.cm.Greys, with_labels=True, node_size=200)
plt.show()


 #ffmpeg -framerate 30 -pattern_type glob -i '*.png' -pix_fmt yuv420p -b:v 4M movie.mp4")