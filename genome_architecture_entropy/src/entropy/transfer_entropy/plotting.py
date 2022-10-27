import numpy as np
import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt
import cmcrameri.cm as cmc


def heatmaps(mat1, mat2=None):
    '''Plots either multivariate TE or both multivariate TE and TE asym as heatmaps'''
    if mat2 is not None:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
        sns.set_theme(style='white')
        plt.rcParams['figure.dpi'] = 75

        plot1 = sns.heatmap(mat1,
                    ax=ax[0],
                    #mask=np.ma.getmask(te_mask),
                    cmap=cmc.cork_r,
                    robust=True,
                    square=True,
                    center=0,
                    linewidths=0,
                    cbar_kws={"shrink": .6})
        plot1.invert_yaxis()

        plot2 = sns.heatmap(mat2,
                    ax=ax[1],
                    #mask=np.ma.getmask(te_mask),
                    cmap=cmc.cork_r,
                    robust=True,
                    square=True,
                    center=0,
                    linewidths=0,
                    cbar_kws={"shrink": .6})
        plot2.invert_yaxis()

    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sns.set_theme(style='white')
        plt.rcParams['figure.dpi'] = 75

        plot1 = sns.heatmap(mat1,
                    ax=ax,
                    #mask=np.ma.getmask(te_mask),
                    cmap=cmc.cork_r,
                    robust=True,
                    square=True,
                    center=0,
                    linewidths=0,
                    cbar_kws={"shrink": .82})
        plot1.invert_yaxis()

    plt.show()


def network(te_asym, coords_model):
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

    # color drive - receive, centered around 0
    drivers = np.sum(te_asym, axis=1)
    receivers = np.sum(te_asym, axis=0)
    vmax_interact = max(np.abs(drivers - receivers))
    node_color_norm = (drivers - receivers) / vmax_interact
    node_list = list(node_color_norm)

    # size = number of total interactions
    n_interactions = np.apply_along_axis(np.sum, 1, te_asym > 0) + np.apply_along_axis(np.sum, 0, te_asym > 0)

    plt.rcParams["figure.figsize"] = (7,7)
    plt.rcParams['figure.dpi'] = 70
    plt.plot(coords_model[0], coords_model[1], '-k') 
    nodes = nx.draw_networkx_nodes(H, 
                                    pos=pos, 
                                    node_size=600,
                                    #node_size=35*n_interactions,
                                    node_color=node_list, 
                                    cmap=cmc.cork_r,
                                    vmax=1.4,
                                    vmin=-1.4)

    edges = nx.draw_networkx_edges(H,
                                    pos=pos,
                                    node_size=600,
                                    #node_size=40*n_interactions,
                                    edge_color=weights_list, 
                                    width=1.8, 
                                    edge_cmap=plt.cm.BuPu,
                                    arrowstyle="->",
                                    arrowsize=20,
                                    min_source_margin=15,
                                    min_target_margin=10,
                                    connectionstyle='arc3,rad=0.2')
    M = H.number_of_edges()
    for i in range(M):
        edges[i].set_alpha(weights_list[i])

    nx.draw_networkx_labels(H, pos=pos)
    plt.show()