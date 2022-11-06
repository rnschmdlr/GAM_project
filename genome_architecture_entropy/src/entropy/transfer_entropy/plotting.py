import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc

cmap = cmc.cork_r(np.linspace(0.05, 0.95, 256))
global my_cmap; my_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', cmap)


def heatmaps(mat1, mat2=None, *args):
    '''Plots either multivariate TE or both multivariate TE and TE asym as heatmaps'''
    sns.set_theme(style='whitegrid')
    n_loci = mat1.shape[0]
    xticks = [str(i) if i % 5 == 0 else ' ' for i in range(1, n_loci+1)]
    yticks = [str(i) if i % 5 == 0 else ' ' for i in range(1, n_loci+1)]

    if mat2 is not None:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))#, sharex=True, sharey=True)
        sns.set_theme(style='white')
        plt.rcParams['figure.dpi'] = 75

        plot1 = sns.heatmap(mat1,
                    ax=ax[0],
                    #mask=np.ma.getmask(te_mask),
                    cmap=my_cmap,
                    robust=True,
                    square=True,
                    center=0,
                    linewidths=0,
                    cbar_kws={"shrink": .75},
                    xticklabels=xticks,
                    yticklabels=yticks,
                    vmin=mat1.min(),
                    vmax=mat1.max(),)
        plot1.invert_yaxis()

        plot2 = sns.heatmap(mat2,
                    ax=ax[1],
                    #mask=np.ma.getmask(te_mask),
                    cmap=my_cmap,
                    robust=True,
                    square=True,
                    center=0,
                    linewidths=0,
                    cbar_kws={"shrink": .75},
                    xticklabels=xticks,
                    yticklabels=yticks,
                    vmin=mat2.min(),
                    vmax=mat2.max(),)
        plot2.invert_yaxis()

    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sns.set_theme(style='white')
        plt.rcParams['figure.dpi'] = 75

        plot1 = sns.heatmap(mat1,
                    ax=ax,
                    #mask=np.ma.getmask(te_mask),
                    cmap=my_cmap,
                    robust=True,
                    square=True,
                    center=0,
                    linewidths=0,
                    cbar_kws={"shrink": .75},
                    xticklabels=xticks,
                    yticklabels=yticks,
                    vmin=mat1.min(),
                    vmax=mat1.max(),)
        plot1.invert_yaxis()
    
    if args:
        plt.savefig(args[0], dpi=150)

    plt.show()


def network(te_asym, coords_model, size, *args):
    '''Transfer entropy network'''
    #te_asym[14:,] = 0
    #te_asym[:14,] = 0
    #te_asym[:,10:] = 0
    #te_asym[:25,] = 0

    mean = np.mean(te_asym[te_asym > 0])
    #te_asym[te_asym < 0.85*np.max(te_asym)] = 0
    te_asym[te_asym < mean] = 0

    H = nx.DiGraph(te_asym)
    #coords_model = series[4].T
    pos = coords_model.T#[1:14]

    # transfer arrows
    edges, weights = zip(*nx.get_edge_attributes(H,'weight').items())
    vmax_w = np.nanmax(list(weights))
    vmin_w = np.nanmin(weights)
    weights_norm = (np.array(weights) - vmin_w) / (vmax_w - vmin_w)
    weights_list = list(np.nan_to_num(weights_norm, copy=False, nan=0.0))

    #n = 10 # n highest interactions
    #idx = np.argsort(te_asym)[:,-n:]
    #data_ = np.zeros(te_asym.shape)
    #for i in range(te_asym.shape[0]):
    #    data_[i,idx[i]] = te_asym[i,idx[i]]
    #data_ = np.nan_to_num(data_, 0)

    # color = drive - receive, centered around 0
    drivers = np.sum(te_asym, axis=1)
    receivers = np.sum(te_asym, axis=0)
    vmax_interact = max(np.abs(drivers - receivers))
    node_color_norm = (drivers - receivers) / vmax_interact
    node_list = list(node_color_norm)

    # size = number of total interactions
    #n_interactions = np.apply_along_axis(np.sum, 1, te_asym > 0) + np.apply_along_axis(np.sum, 0, te_asym > 0)

    plt.plot(coords_model[0], coords_model[1], '-k') 
    nodes = nx.draw_networkx_nodes(H, 
                                    pos=pos, 
                                    node_size=800,
                                    #node_size=300+50*n_interactions,
                                    node_color=node_list, 
                                    cmap=my_cmap,
                                    vmax=1.4,
                                    vmin=-1.4)

    edges = nx.draw_networkx_edges(H,
                                    pos=pos,
                                    node_size=800,
                                    #node_size=300+50*n_interactions,
                                    edge_color=weights_list, 
                                    width=2, 
                                    edge_cmap=plt.cm.BuPu,
                                    arrowstyle="->",
                                    arrowsize=20,
                                    min_source_margin=15,
                                    min_target_margin=10,
                                    connectionstyle='arc3,rad=0.2')
    M = H.number_of_edges()
    for i in range(M):
        edges[i].set_alpha(weights_list[i])

    
    nx.draw_networkx_labels(H, pos=pos, font_weight='bold', font_size=11, labels={i: i+1 for i in range(len(pos))})

    plt.axis('off')
    plt.rcParams["figure.figsize"] = size
    plt.rcParams['figure.dpi'] = 70

    if args:
        plt.savefig(args[0], dpi=150)
    
    plt.show()
    


def progression(coords_model, matrices_vmax_dict, drivers, norm, state_H, *args):
    # style and colors
    sns.set_theme(style='whitegrid')
    cmap_legend_scatter = plt.cm.ScalarMappable(norm=mcolors.Normalize(-1, 1), cmap=my_cmap)
    #cmap_legend_scatter.set_clim(vmin=-1.4, vmax=1.4)

    # dynamic coordinate boundaries
    xmin = np.min(coords_model[0])
    xmax = np.max(coords_model[0])
    width = xmax - xmin
    #length = coords_model.shape[1]
    ymin = np.min(coords_model[1])
    ymax = np.max(coords_model[1])
    height = ymax - ymin
    scaling = width / height # for scatter point size

    # set subtitle text
    text_info = ''
    for i in state_H:
        text_info = text_info + i + str(state_H[i])

    # set figure
    fig = plt.figure(figsize=(14, 4))
    plt.figtext(x=0.055, y=0.87, s=text_info, fontsize=12, weight='bold')
    gs = gridspec.GridSpec(2, 6, width_ratios=[.07,1,1,1,1,1])
    ax1 = plt.subplot(gs[0:2, 1:4])
    ax2 = plt.subplot(gs[0:2, 4:6])
    ax3 = plt.subplot(gs[0:2, 0:1]) # colorbar scatter
    axes = [ax1, ax2, ax3]
    
    # set coordinate plot
    axes[0].plot(coords_model[0], coords_model[1], '-k', linewidth=3-0.3*scaling,  zorder=1) 
    axes[0].scatter(coords_model[0], coords_model[1], s=350-35*scaling, c=drivers, cmap=my_cmap, norm=norm, zorder=2)#, vmax=0.53)
    axes[0].set_aspect('equal')
    axes[0].tick_params(labelleft=False, labelbottom=False)
    #axes[0].set_title('Driving loci in chromatin model (net positive transfer entropy)', loc='center', fontsize=14, y=1.1)
    axes[0].axis('off')
    plt.colorbar(cmap_legend_scatter, cax=axes[2], shrink=0.3).outline.set_visible(False)

    axes[2].yaxis.tick_left()
    # set y-axis label from -1 to 1 in 0.5 steps
    axes[2].set_yticks(np.arange(-1, 1.1, 0.5))
    axes[2].set_ylabel('net transfered entropy \n proportion to max %', labelpad=-80)

    # extend xlim and ylim
    extend = -0.07
    xlim = axes[0].get_xlim()
    xrange = xlim[1] - xlim[0]
    ylim = axes[0].get_ylim()
    yrange = ylim[1] - ylim[0]
    axes[0].set_xlim(xlim[0] + extend*xrange, xlim[1] - extend*xrange)
    axes[0].set_ylim(ylim[0] + extend*yrange, ylim[1] - extend*yrange)

    # set the heatmap plots
    for index, (key, value) in enumerate(matrices_vmax_dict.items()):
        #axes[index+1].set_title(key, loc='left', fontsize=13)
        axes[index+1].tick_params(labelleft=False, labelbottom=False)

        plot = sns.heatmap(value[0],
                    ax=axes[index+1],
                    mask=None,
                    cmap=my_cmap,
                    square=True,
                    linewidths=0,
                    center=0,
                    vmax=value[1],
                    vmin=-value[1])
        plot.invert_yaxis()

    #plt.tight_layout()
    if args:
        plt.savefig(args[0], dpi=150)

    plt.show()