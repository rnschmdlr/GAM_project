import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
import matplotlib as mpl

cmap = cmc.cork_r(np.linspace(0.15, 0.85, 256))
global my_cmap; my_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', cmap)
blue = cmc.cork_r(np.linspace(0.5, 0.85, 256))
global my_cmap_blue; my_cmap_blue = mcolors.LinearSegmentedColormap.from_list('blue', blue)


def heatmaps(mat1, vmax1, vmax2, mat2=None, mat3=None, *args):
    '''Plots either multivariate TE or both multivariate TE and TE asym as heatmaps'''
    #sns.set_theme(style='whitegrid')
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="whitegrid", rc=custom_params)
    sns.set_context('paper', font_scale=2)
    plt.rcParams['figure.dpi'] = 75
    cbar = False

    n_loci = mat1.shape[0]
    if n_loci < 20:
        xticks = [str(i) if i % 5 == 0 or i==1 else '•' for i in range(1, n_loci+1)]
        yticks = [str(i) if i % 5 == 0 or i==1 else '•' for i in range(1, n_loci+1)]
    else:
        xticks = ''
        yticks = ''

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    plot1 = sns.heatmap(mat1,
                    ax=ax[0],
                    #mask=np.ma.getmask(te_mask),
                    cmap=my_cmap,
                    robust=True,
                    square=True,
                    center=0,
                    linewidths=0,
                    cbar=cbar,
                    cbar_kws={"shrink": .75},
                    xticklabels=xticks,
                    yticklabels=yticks,
                    vmin=0,
                    vmax=vmax1)
    ax[0].tick_params(axis='x', rotation=0)
        
    plot1.invert_yaxis()

    plot1.axhline(y=0, color='gray',linewidth=2)
    plot1.axhline(y=n_loci, color='gray',linewidth=2)
    plot1.axvline(x=0, color='gray',linewidth=2) #check
    plot1.axvline(x=n_loci-0, color='gray',linewidth=2)

    plot1.set_xlabel('Locus $i$')
    plot1.set_ylabel('Locus $j$')

    if mat2 is not None:
        plot2 = sns.heatmap(mat2,
                    ax=ax[1],
                    #mask=np.ma.getmask(te_mask),
                    cmap=my_cmap,
                    robust=True,
                    square=True,
                    center=0,
                    linewidths=0,
                    cbar=cbar,
                    cbar_kws={"shrink": .75, "location": 'bottom', "label": "Asymmetric transfer entropy"},
                    xticklabels=xticks,
                    yticklabels=[],
                    vmin=-vmax2,
                    vmax=vmax2)
        ax[1].tick_params(axis='x', rotation=0)

        plot2.invert_yaxis()
        plot2.axhline(y=0, color='gray',linewidth=2)
        plot2.axhline(y=n_loci, color='gray',linewidth=2)
        plot2.axvline(x=0, color='gray',linewidth=2) #check
        plot2.axvline(x=n_loci-0, color='gray',linewidth=2)

        plot2.set_xlabel('Locus $i$')
    else: 
        mat2 = np.zeros_like(mat1)
        ax[2].axis('off')
        
    if mat3 is not None:
        plot3 = sns.heatmap(mat3,
                    ax=ax[2],
                    #mask=np.ma.getmask(te_mask),
                    cmap=my_cmap,
                    square=True,
                    center=0,
                    linewidths=0,
                    cbar=cbar,
                    cbar_kws={"shrink": .75, "location": 'bottom', "label": "Asymmetric transfer entropy"},
                    xticklabels=xticks,
                    yticklabels=[])
        ax[2].tick_params(axis='x', rotation=0)

        plot3.invert_yaxis()
        plot3.axhline(y=0, color='gray',linewidth=2)
        plot3.axhline(y=n_loci, color='gray',linewidth=2)
        plot3.axvline(x=0, color='gray',linewidth=2) #check
        plot3.axvline(x=n_loci-0, color='gray',linewidth=2)

        plot3.set_xlabel('Locus $i$')
    else: 
        mat3 = np.zeros_like(mat1)
        ax[2].axis('off')

    #adjust spacig between subplots
    fig.subplots_adjust(wspace=0.05)

    if args:
        plt.savefig(args[0], dpi=120, bbox_inches="tight")

    plt.show()


def network(te_asym, coords_model, node_list, *args):
    '''Transfer entropy network'''
    xmin = np.min(coords_model[0])
    xmax = np.max(coords_model[0])
    width = xmax - xmin
    ymin = np.min(coords_model[1])
    ymax = np.max(coords_model[1])
    height = ymax - ymin
    plt.rcParams["figure.figsize"] = (1.5*width, 1.5*height)

    cutoff = np.percentile(te_asym[te_asym > 0], 75)
    te_asym[te_asym < cutoff] = 0

    H = nx.DiGraph(te_asym)
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

    # size = number of total interactions
    #n_interactions = np.apply_along_axis(np.sum, 1, te_asym > 0) + np.apply_along_axis(np.sum, 0, te_asym > 0)

    plt.plot(coords_model[0], coords_model[1], '-k', linewidth=4) 
    nodes = nx.draw_networkx_nodes(H, 
                                    pos=pos, 
                                    node_size=1700,
                                    #node_size=300+50*n_interactions,
                                    node_color=node_list, 
                                    cmap=my_cmap,
                                    vmax=1.3,
                                    vmin=-1.3,
                                    edgecolors='grey',
                                    linewidths=1,)

    edges = nx.draw_networkx_edges(H,
                                    pos=pos,
                                    node_size=1700,
                                    #node_size=300+50*n_interactions,
                                    edge_color=weights_list, 
                                    width=3, 
                                    edge_cmap=plt.cm.BuPu,
                                    arrowstyle="->",
                                    arrowsize=30,
                                    min_source_margin=22,
                                    min_target_margin=10,
                                    connectionstyle='arc3,rad=0.3')
    M = H.number_of_edges()
    for i in range(M):
        edges[i].set_alpha(weights_list[i])
        #edges[i].set_zorder(3)
    
    nx.draw_networkx_labels(H, pos=pos, font_size=18, labels={i: i+1 for i in range(len(pos))})

    plt.axis('off')
    plt.rcParams['figure.dpi'] = 70

    if args:
        plt.savefig(args[0], dpi=300, bbox_inches='tight')
    
    plt.show()

    # center value between cutoff and max
    mean = (cutoff + vmax_w) / 2
    halfway1 = (cutoff + mean) / 2
    halfway2 = (mean + vmax_w) / 2
    #generate_colorbar('Net transfer entropy (bits)', [cutoff, halfway1, mean, halfway2, vmax_w], [round(cutoff, 2), round(halfway1, 2), round(mean, 2), round(halfway2, 2), round(vmax_w, 2)], 'h', 'bottom', (5, 0.3), plt.cm.BuPu)
    generate_colorbar('Net transfer entropy (bits)', [cutoff, mean, vmax_w], [round(cutoff, 2), round(mean, 2), round(vmax_w, 2)], 'h', 'bottom', (5, 0.3), plt.cm.BuPu)
    #generate_colorbar('Net transfer entropy', [mean, 0.09, 0.12, vmax_w], [round(mean, 2), 0.09, 0.12, round(vmax_w, 2)], 'v', 'left', (0.3, 4), plt.cm.BuPu)

    
def coords(coords_model, influence, *args):
    # style and colors
    sns.set_theme(style='white', rc={"axes.spines.right": False, "axes.spines.top": False, "axes.spines.left": False, "axes.spines.bottom":False})
    plt.rcParams['figure.dpi'] = 100
    norm = mcolors.Normalize(-1,1)

    # dynamically set point size and line width
    xmax = np.max(coords_model[0])
    xmin = np.min(coords_model[0])
    ymax = np.max(coords_model[1])
    ymin = np.min(coords_model[1])
    yrange = np.abs(ymax - ymin)
    xrange = np.abs(xmax - xmin)
    if xrange/yrange > 1:
        point_size = 300**2 / ((xrange)**2 * yrange/xrange)
        line_width = 80 / xrange
    else :
        point_size = 250**2 / ((yrange)**2)
        line_width = 40 / yrange
    point_size = 300
    line_width = 5

    # set coordinate plot
    fig, ax = plt.subplots()#figsize=(16,8))
    ax.plot(coords_model[0], coords_model[1], '-k', linewidth=line_width,  zorder=1) 
    #ax.scatter(coords_model[0], coords_model[1], s=point_size, c=influence, cmap=my_cmap, norm=norm, linewidths = 0.5, edgecolor='grey', clip_on=False, zorder = 10)
    ax.scatter(coords_model[0], coords_model[1], s=point_size, c='k', clip_on=False, zorder = 10)
    ax.set_aspect('equal')
    ax.tick_params(labelleft=False, labelbottom=False)

    # recompute the ax.dataLim
    #ax.relim()
    # update ax.viewLim using the new dataLim
    #ax.autoscale_view()

    #extend axis
    #ax.set_xlim(xmin-1, xmax+1)
    #ax.set_ylim(ymin-1, ymax+1)
    
    if args:
        plt.savefig(args[0], dpi=150, bbox_inches='tight')

    plt.show()



def progression(coords_model, xylim, drivers, state_H, *args):
    # style and colors
    sns.set_theme(style='white', rc={'axes.edgecolor':'grey'})
    plt.rcParams['figure.dpi'] = 100
    norm = mcolors.Normalize(-1,1)
    cmap_legend_scatter = plt.cm.ScalarMappable(norm=norm, cmap=my_cmap)
    point_size = 270 / ((xylim[0] - xylim[1])**2.1) * 150
    line_width = 3 / ((xylim[0] - xylim[1])) * 10

    # set subtitle text
    text_info = ''
    for i in state_H:
        text_info = text_info + i + str(state_H[i])

    # set figure
    fig = plt.figure(figsize=(6, 4))
    plt.figtext(x=0.51, y=0.87, fontsize=12, weight='bold', s=text_info)
    gs = gridspec.GridSpec(2, 3, width_ratios=[.08,1,1])
    ax1 = plt.subplot(gs[0:2, 1:3])
    ax2 = plt.subplot(gs[0:2, 0:1]) # colorbar scatter
    axes = [ax1, ax2]
    
    # set coordinate plot
    axes[0].plot(coords_model[0], coords_model[1], '-k', linewidth=line_width,  zorder=1) 
    axes[0].scatter(coords_model[0], coords_model[1], s=point_size, c=drivers, cmap=my_cmap, norm=norm, linewidths = 0.5, edgecolor='grey', zorder=2)
    axes[0].set_aspect('equal')
    axes[0].tick_params(labelleft=False, labelbottom=False)
    #axes[0].set_xlabel('x', loc='right', fontsize=11)
    #axes[0].set_ylabel('y', loc='top', fontsize=11)
    #axes[0].axis('off')

    # set colorbar
    plt.colorbar(cmap_legend_scatter, cax=axes[1], shrink=0.1).outline.set_visible(False)
    axes[1].yaxis.tick_left()
    axes[1].set_yticks(np.arange(-1, 1.1, 0.5))
    axes[1].set_ylabel('Net influence % of max', labelpad=-55)

    # extend xlim and ylim
    extend = -0.1
    xrange = xylim[1] - xylim[0]
    yrange = xylim[3] - xylim[2]
    axes[0].set_xlim(xylim[0] + extend*xrange, xylim[1] - extend*xrange)
    axes[0].set_ylim(xylim[2] + 2, xylim[3] - 2)
    
    if args:
        plt.savefig(args[0], dpi=300)

    plt.show()



def net_influence_bars(df, *args):
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=2)
    fig, ax = plt.subplots(figsize=(24,12)) #(16,8))

    # Normalize values for color map
    norm = plt.Normalize(-1,1)

    # set barplot
    ax = sns.barplot(data=df, x='Locus', y='T_A positive', linewidth=2, facecolor=(0, 0, 0, 0), edgecolor=my_cmap(norm(df['T_A positive'])), orient='v', zorder=1, hatch='//')
    ax = sns.barplot(data=df, x='Locus', y='T_A negative', linewidth=2, facecolor=(0, 0, 0, 0), edgecolor=my_cmap(norm(df['T_A negative'])), orient='v', zorder=1, hatch='//')
    ax = sns.barplot(data=df, x='Locus', y='T_A net', linewidth=2, edgecolor=my_cmap(norm(df['T_A net'])), palette=my_cmap(norm(df['T_A net'])), orient='v', zorder=2)

    ax.set_ylim(-1, 1)
    ax.set_ylabel('Influence % of max', fontsize=24, labelpad =10)
    ax.set_xlabel('Locus', fontsize=24)
    #plt.legend(labels = ['absolute', 'net'])
    #sns.move_legend(ax, "upper center", bbox_to_anchor=(1, 1.25))

    # set lines at mean tan and mean tap
    #plt.axvline(x=df.mean(axis=0)[0], color='k', linestyle='--')
    #plt.axvline(x=np.mean(tan), color='k', linestyle='--')

    n_loci = len(df)
    if n_loci > 30:
        xticks = [str(i) if i % 10 == 0 or i==1 else '' for i in range(1, n_loci+1)]  
    else:
        xticks = [str(i) if i % 5 == 0 or i==1 else '' for i in range(1, n_loci+1)]

    plt.xticks(np.arange(0, n_loci, 1), xticks, fontsize = 20)
    plt.yticks(fontsize = 20)

    minor = [i-1 for i in range(1, n_loci) if i % 5 == 0] 
    for i in range(len(minor)):
        plt.axvline(x=minor[i], color='lightgrey', linestyle='-', linewidth=0.5, zorder=0)

    major = [i-1 for i in range(1, n_loci) if i % 10 == 0]
    for i in range(len(major)):
        plt.axvline(x=major[i], color='lightgrey', linestyle='-', linewidth=1, zorder=0)

    if args:
        plt.savefig(args[0], dpi=300, bbox_inches='tight')

    plt.show()



    
def te_significance(pvalues, *args):
    n_loci = len(pvalues)

    sns.set_context('paper', font_scale=2)
    cmap = sns.color_palette("coolwarm_r", 6)
    new_palette = cmap[:1] + cmap[-3:]
    #register new colormap
    cm = mcolors.LinearSegmentedColormap.from_list('significance', new_palette, N=4)

    xticks = [str(i) if i % 5 == 0 or i==1 else '•' for i in range(1, n_loci+1)]
    yticks = [str(i) if i % 5 == 0 or i==1 else '•' for i in range(1, n_loci+1)]

    ax = sns.heatmap(pvalues, cmap=cm, square=True, cbar=False, vmin=1, vmax=3, xticklabels=xticks, yticklabels=yticks,
                        cbar_kws={"location": 'bottom', "label": "Significance level"})
    
    #colorbar = ax.collections[0].colorbar
    #colorbar.set_ticks([0.5, 1.5, 2.5, 3.5])
    #colorbar.set_ticklabels(['NS', 'p < 0.05', 'p < 0.01', 'p < 0.001'])

    ax.invert_yaxis()
    ax.axhline(y=0, color='gray',linewidth=1)
    ax.axhline(y=n_loci, color='gray',linewidth=1)
    ax.axvline(x=0, color='gray',linewidth=1)
    ax.axvline(x=n_loci-0, color='gray',linewidth=1)
    ax.set_xlabel('Locus $i$')
    ax.set_ylabel('Locus $j$')
                        
    plt.rcParams["figure.figsize"] = (8, 8)
    plt.rcParams['figure.dpi'] = 100   

    if args:
        plt.savefig(args[0], dpi=150)

    plt.show()

    generate_colorbar('Significance level', [1,2,3,4], ['NS', 'p < 0.05', 'p < 0.01', 'p < 0.001'], 'h', 'bottom', (4, 0.5), cmap=cm)  



def generate_colorbar(label, ticks, tick_labels, orient, loc, size, cmap, *args):
    """
    Function to generate a colorbar.

    Parameters:
    - label: The label for the colorbar.
    - ticks: An array-like with the colorbar tick locations.
    - tick_labels: A list or array-like with labels for the colorbar ticks.
    - orientation: The orientation of the colorbar, either 'vertical' or 'horizontal'.
    - size: A tuple specifying the size of the colorbar figure.
    - cmap: A colormap instance or registered colormap name.
    """
    plt.rcParams['figure.dpi'] = 150
    fig, ax = plt.subplots(figsize=size)
    #fig.subplots_adjust(bottom=0.7)

    # Ensure the colorbar has correct orientation
    if orient not in ['v', 'h']:
        raise ValueError("orientation must be either 'vertical' or 'horizontal'")
    if orient == 'v':
        orientation = 'vertical'
    elif orient == 'h':
        orientation = 'horizontal'

    if cmap == 'blue':
        cmap = my_cmap_blue
    elif cmap == 'green-blue':
        cmap = my_cmap

    norm = mpl.colors.Normalize(vmin=min(ticks), vmax=max(ticks))
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax, 
             orientation=orientation, 
             label=label, 
             ticks=ticks,
             ticklocation=loc)
    cbar.outline.set_visible(False)
    cbar.set_label(label, size=24)

    if orientation == 'vertical':
        cbar.ax.set_yticklabels(tick_labels, size=20)
    else:
        cbar.ax.set_xticklabels(tick_labels, size=20)

    plt.rcParams['figure.dpi'] = 150
    if args:
        plt.savefig(args[0], dpi=300, bbox_inches='tight')
    plt.show()



def trajectories(df, loci, *args):
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=2)
    fig, ax = plt.subplots(figsize=(8,6))
    palette = ['#9A3C4D', '#3D939B', '#654D9A', '#B47141', ]
    df_selected = df[df['Locus'].isin(loci)]

    ax = sns.lineplot(data=df_selected, x='Timepoint', y='Influence % of max', hue='Locus', style='Influence', palette=palette[:len(loci)], linewidth=2.5)
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2)
    
    if args:
        plt.savefig(args[0], dpi=300, bbox_inches='tight')
    plt.show()
