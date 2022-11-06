# %% import
'''Segregation Analysis'''
import os
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from tqdm.auto import tqdm
import cmcrameri.cm as cmc

os.chdir('/Users/pita/Documents/Rene/GAM_project/genome_architecture_entropy/src/')

import entropy.shannon_entropies.compute_2d_entropies as em
import entropy.transfer_entropy.compute_transfer_entropy as te
import entropy.transfer_entropy.transition as tb
import toymodel.sampling


# %%
def plot_large_ensemble(coords_model, matrices_vmax_dict, mi_sums_vmax, state_H, *args):
    '''4x2 plot fxn'''
    # Generate a mask for the upper triangle from value shape, k=1 to see diagonal line in heatmap
    #any_mat = matrices_vmax_dict['Norm. Mutual Information (NMI)'][0]
    #mask_shape = (any_mat.shape[1], any_mat.shape[1])
    #mask = np.triu(np.ones(mask_shape, dtype=bool), k=0)

    # style and colors
    sns.set_theme(style='dark')
    cmap_scatter_ = cmc.cork_r(np.linspace(0.5, 1, 256))
    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap', cmap_scatter_)
    cmap_legend_scatter = plt.cm.ScalarMappable(cmap=my_cmap)

    norm1 = matplotlib.colors.Normalize(-1, 1)
    cmap_legend_heatmap = plt.cm.ScalarMappable(norm=norm1, cmap=cmc.cork_r)

    # normalize mi_sums
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=mi_sums_vmax[1])

    # dynamic coordinate boundaries
    xmin = np.min(coords_model[0])
    xmax = np.max(coords_model[0])
    width = xmax - xmin
    length = coords_model.shape[1]
    scaling = length / width # for scatter point size

    # set subtitle text
    text_info = ''
    for i in state_H:
        text_info = text_info + i + str(state_H[i])

    # set figure
    fig = plt.figure(figsize=(10.5, 7))
    #plt.figtext(x=0.87, y=0.95, s=text_info, fontsize=16, weight='bold')
    
    gs = gridspec.GridSpec(4, 7, width_ratios=[1,1,1,1,1,1,0.1])
    ax1 = plt.subplot(gs[0:2, 0:4])
    ax2 = plt.subplot(gs[0:2, 4:6])
    ax3 = plt.subplot(gs[2:4, 0:2])
    ax4 = plt.subplot(gs[2:4, 2:4])
    ax5 = plt.subplot(gs[2:4, 4:6])
    ax6 = plt.subplot(gs[0:2, 6:7]) # colorbar scatter
    ax7 = plt.subplot(gs[2:4, 6:7]) # colorbar heatmaps
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    
    # set coordinate plot
    axes[0].plot(coords_model[0], coords_model[1], '-k', zorder=1) 
    axes[0].scatter(coords_model[0], coords_model[1], s=50*1.5*scaling, c=mi_sums_vmax[0], norm=normalize, cmap=my_cmap, zorder=2)
    axes[0].set_aspect('equal', 'datalim')
    axes[0].tick_params(labelleft=False, labelbottom=False)
    axes[0].set_title(y=1.15, label=text_info, fontsize=16, weight='bold', loc='left')
    plt.figtext(x=0.025, y=0.9, s='Mutual Information of Chromatin Model', fontsize=14)
    axes[0].axis('off')
    plt.colorbar(cmap_legend_scatter, cax=axes[5], shrink=0.3).outline.set_visible(False)
    plt.colorbar(cmap_legend_heatmap, cax=axes[6], shrink=0.5).outline.set_visible(False)

    #making a custom color map
    #colors1 = cmc.oslo_r(np.linspace(0, 1, 128))
    #colors2 = cmc.acton(np.linspace(0, 1, 128))
    #colors = np.vstack((colors2, colors1))
    #mymap = matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap', colors)

    # set the heatmap plots
    for index, (key, value) in enumerate(matrices_vmax_dict.items()):
        axes[index+1].set_title(key, loc='left', fontsize=14)
        axes[index+1].tick_params(labelleft=False, labelbottom=False)
        axes[index+1].invert_yaxis()
        set_cmap = matplotlib.cm.get_cmap(cmc.cork_r).copy()
        #set_cmap.set_under('powderblue')
        #set_cmap.set_over('gold')
        
        plot = sns.heatmap(value[0],
                    ax=axes[index+1],
                    mask=None,
                    cmap=set_cmap,
                    robust=True,
                    square=True,
                    linewidths=0,
                    cbar=False, #cbar_kws=dict(shrink=0),#, extend='max', extendrect=True),
                    vmax=1,
                    vmin=-1)
        plot.invert_yaxis()
        
    plt.tight_layout()
    #plt.savefig(path_series_out + args[0], dpi=120)
    #plt.close(fig)


def scale(a):
    a = a - np.nanmin(a)
    a = a - np.diag(np.diag(a))
    vmax = np.nanmax(a)
    # divide calculation anywhere 'where' a does not equal zero
    a = np.divide(a, vmax, out=np.zeros_like(a), where = vmax != 0)

    return a



# %% 
'''Interpolate Series'''
fps = 30 
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
np.save(path_model + 'toymodel_interpolated.npy', series_extended)



# %%
'''Series 2D Calculations'''
path_model = '/data/toymodels/model2/'
model = 'toymodel2'
series = np.load(path_model + model + '.npy')
path_series_out = path_model + '/series'

n_slice = 2000
#series_extended = np.load(path_model + 'toymodel_interpolated.npy')
seg_mats = np.empty((series.shape[0], n_slice, series.shape[1]))
mi_vmax = 0

# precompute max values over series
for state in tqdm(range(series.shape[0])):
    coords_model = series[state].T
    seg_mat = toymodel.sampling.slice(coords_model, n_slice)

    mi_mat = em.normalized_mutual_information(seg_mat.T, c=False)
    mi_mat = scale(mi_mat)
    mi_sums = np.sum(mi_mat, axis=0)
    mi_vmax = max(mi_sums.max(), mi_vmax)
    
    #ncmi_mat = em.normalized_mutual_information(seg_mat.T)
    #ncmi_mat = np.nan_to_num(ncmi_mat, copy=False, nan=0)
    #for i in range(-1, 2):
    #    ncmi_mat_d = ncmi_mat - np.diag(np.diag(ncmi_mat, k=i), k=i)
    #mask = ncmi_mat < np.mean(ncmi_mat_d)
    #mi_sums = np.sum(ncmi_mat, axis=0, where=np.invert(mask))
    #vmax[2] = max(mi_sums.max(), vmax[2])

# compute all information measures:
for state in tqdm(range(series.shape[0])):
    coords_model = series[state].T
    seg_mat = toymodel.sampling.slice(coords_model, n_slice)

    mi_mat = em.normalized_mutual_information(seg_mat.T, c=False)
    mi_mat = scale(mi_mat)
    mi_sums = np.sum(mi_mat, axis=0)
    np.fill_diagonal(mi_mat, 1)

    ncmi_mat = em.normalized_mutual_information(seg_mat.T)
    ncmi_mat = np.nan_to_num(ncmi_mat, copy=False, nan=0)
    #for i in range(-1, 2):
    #    ncmi_mat_d = ncmi_mat - np.diag(np.diag(ncmi_mat, k=i), k=i)
    #mask = ncmi_mat < np.mean(ncmi_mat_d)
    #mi_sums = np.sum(ncmi_mat, axis=0, where=np.invert(mask))

    coseg_mat = np.ones_like(ncmi_mat) 
    # depreciated
    # ci.dprime_2d(seg_mat.T.astype(int), seg_mat.T.astype(int))

    npmi_mat = em.npmi_2d_fast(seg_mat.T.astype(int), seg_mat.T.astype(int))
    npmi_mat = np.nan_to_num(npmi_mat, copy=False, nan=0)
    
    # prepare plot function call
    matrices_vmax_dict = {'NMI*': [mi_mat, 1],
                          'LD': [coseg_mat, 1],
                          'NPMI': [npmi_mat, 1],
                          'NCMI': [ncmi_mat, 1],
                          }
    mi_sums_vmax = [mi_sums, mi_vmax]
    state_H = {'Timestep ': (1 + state) // 30}
    file = '/state_2_%03d.png' % (state)

    plot_large_ensemble(coords_model, matrices_vmax_dict, mi_sums_vmax, state_H, file)

    #ffmpeg -framerate 30 -pattern_type glob -i '*.png' -pix_fmt yuv420p -b:v 4M movie.mp4")
