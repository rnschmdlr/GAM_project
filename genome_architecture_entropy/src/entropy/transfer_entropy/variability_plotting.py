import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors


default = sns.cubehelix_palette(start=0.5, rot=-1.5, light=0, dark=1, as_cmap=True, gamma=0.9, hue=1.25)
tulip = sns.cubehelix_palette(start=2.5, rot=-1.1, light=0, dark=1, as_cmap=True, gamma=0.75, hue=1.2)
dusk_blue = sns.cubehelix_palette(rot=-.15, light=1, as_cmap=True) 
dusk_blue_r = sns.cubehelix_palette(rot=-.15, light=1, as_cmap=True, reverse=True)
magic = sns.cubehelix_palette(start=2.5, rot=0.8, light=0, dark=1, as_cmap=True, gamma=0.95, hue=1.5)
alaska = sns.cubehelix_palette(start=0.7, rot=-1.1, light=0.05, dark=0.95, as_cmap=True, gamma=1, hue=1.35)
alaska2 = sns.cubehelix_palette(start=0.7, rot=-1.34, light=0.1, dark=0.85, as_cmap=True, gamma=0.85, hue=1.25)
alaska3 = sns.cubehelix_palette(start=0.83, rot=-1.28, light=0, dark=1, as_cmap=True, gamma=0.75, hue=1.3)
global mcmap; mcmap = alaska3


#cmaps_list = ['RdYlBu_r', alaska3, 'viridis']
#for cmap in cmaps_list:
#    if type(cmap) == str: cmap = plt.get_cmap(cmap)
#    global mcmap; mcmap = cmap
#    visualize_variation(variation_matrix, None, mode, name)

sns.set_context('paper', font_scale=1.9)
dpi=150


def vector_difference(method_label, matrix, vmax=None, mode=None):
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8), dpi=dpi)
    plt.title(f"{method_label} difference between {mode} vectors of loci co-sampling")
    plt.imshow(matrix, cmap=mcmap, interpolation='none', vmax=vmax)
    plt.colorbar(label=method_label)
    plt.xlabel('Locus')
    plt.ylabel('Locus')
    plt.grid(False)
    plt.show()


def variability(matrix, title=None, unit=None, interesting_pairs=None, vmax=None, vmin=None, ratio=False, center=None):
    # fill upper triangular with NaNs
    tril = np.tril_indices(matrix.shape[0], k=0)
    matrix[tril] = np.nan

    cmap = mcmap

    # special ratio mode
    if ratio == True:
        vmax = np.nanpercentile(matrix, 99)
        vmin = np.nanpercentile(matrix, 1)  

        # shifted ratio center for norm and colormap
        if center == None:
            center = (vmax + vmin) / 2
            print(f"Values centered around: {round(center, 2)}")

        end, start = map_values(vmax, center, vmin)

        # Create a resampled colormap
        cmap_redblue = plt.get_cmap('RdBu_r')
        cmap = cmap_redblue(np.linspace(start, end, 256))
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', cmap)

    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8), dpi=dpi)
    plt.title(title, pad=10, loc='center', fontsize=16)
    plt.imshow(matrix, cmap=cmap, interpolation='none', vmax=vmax, vmin=vmin)
    plt.colorbar(label=unit, pad=0.1, shrink=0.4, aspect=10, orientation='horizontal')
    plt.clim(vmin, vmax)

    # turn off ticks and labels
    plt.tick_params(axis='both', which='both', bottom=True, top=False, left=False, right=False,
                    labelbottom=True, labeltop=False, labelleft=False, labelright=False)
    
    # put white line on diagonal
    #plt.plot([0, matrix.shape[0]], [0, matrix.shape[0]], color='white', linewidth=4, linestyle="-", clip_on=False)

    # remove border around plot
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    # Highlight interesting pairs if provided
    if interesting_pairs is not None:
        for (row, col) in interesting_pairs:
            plt.text(col, row, '+', ha='center', va='center', color='yellow', fontsize=16, weight='bold',
                     path_effects=[pe.withStroke(linewidth=2, foreground="black")])
            
    plt.grid(False)
    plt.show()


def variability_difference(matrix, mode=None, operation=None, mid=None):
    # robust min and max
    if mid is None:
        vmax = np.nanpercentile(matrix, 99)
        vmin = np.nanpercentile(matrix, 1)
        # Create a TwoSlopeNorm normalization
        midpoint = 0.5 * (vmax + vmin)
    else:
        midpoint = mid
        vmax = np.nanmax(matrix) #np.nanpercentile(matrix, 99)
        vmin = np.nanmin(matrix) #np.nanpercentile(matrix, 1)
        max_abs = max(np.abs(vmax), np.abs(vmin))
        vmax = max_abs
        vmin = -max_abs

    norm = TwoSlopeNorm(vmin=vmin, vcenter=midpoint, vmax=vmax)

    # style
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8), dpi=dpi)
    plt.title(f"{operation} of {mode} vectors")
    plt.imshow(matrix, cmap='RdBu_r', interpolation='none', norm=norm)
    plt.colorbar(label=operation)
    plt.xlabel('Locus')
    plt.ylabel('Locus')
            
    plt.grid(False)
    plt.show()


def contact_map(npmi_matrix, dataset_str):
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8), dpi=dpi)
    plt.title(dataset_str + " dataset NPMI Matrix")
    plt.imshow(npmi_matrix, cmap=mcmap, interpolation='none', vmin=0, vmax=1)
    plt.colorbar(label='NPMI')
    plt.xlabel('Locus')
    plt.ylabel('Locus')
    plt.grid(False)
    plt.show()



def ridge(prob_vectors, centers, mode, labels=None, pair_colors=None, npmi1=None, npmi2=None, npmi3=None, pm_ratio=None):
    sns.set_theme(style="dark", rc={"axes.facecolor": (0, 0, 0, 0), 'figure.facecolor': 'black',})

    def legend():
        # Create a separate figure for the legend
        legend_fig, legend_ax = plt.subplots(figsize=(2,2))

        # Manually create the legend items as shorter vertical lines
        x_pos = 0
        line_height = 0.1
        label_x_pos = 0.02

        # Create vertical lines and labels for each legend item
        #legend_ax.plot([x_pos, x_pos], [0.1, 0.1 + line_height], color='lightgrey', linewidth=2, linestyle="-.", label='Maternal NPMI')
        #legend_ax.text(label_x_pos, 0.1 + line_height / 2, 'Maternal NPMI', verticalalignment='center', fontsize=16, color='white')

        #legend_ax.plot([x_pos, x_pos], [0.3, 0.3 + line_height], color='lightgrey', linewidth=2, linestyle=":", label='Paternal NPMI')
        #legend_ax.text(label_x_pos, 0.3 + line_height / 2, 'Paternal NPMI', verticalalignment='center', fontsize=16, color='white')

        #legend_ax.plot([x_pos, x_pos], [0.5, 0.5 + line_height], color='lightgrey', linewidth=2, linestyle="-", label='Combined NPMI')
        #legend_ax.text(label_x_pos, 0.5 + line_height / 2, 'Combined NPMI', verticalalignment='center', fontsize=16, color='white')

        legend_ax.plot([x_pos, x_pos], [0.7, 0.7 + line_height + 0.05], color='red', linewidth=2, linestyle="-", label='Trimmed Mean')
        legend_ax.text(label_x_pos, 0.7 + line_height / 2, 'Mean', verticalalignment='center', fontsize=16, color='white')

        legend_ax.plot([x_pos, x_pos], [0.9, 0.9 + line_height], color='dodgerblue', linewidth=2, linestyle="--", label='Trimmed Mean')
        legend_ax.text(label_x_pos, 0.9 + line_height / 2, 'RE (P, M)', verticalalignment='center', fontsize=16, color='white')

        # Remove the axis from the legend figure
        legend_ax.axis('off')

        # Display the legend figure
        plt.show()


    # Get the number of vectors
    n_vectors = len(prob_vectors)
    
    # Convert data into the required DataFrame format
    x = np.concatenate(prob_vectors)
    if labels is None:
        labels = [f"Pair {i}" for i in range(n_vectors)]
    g = np.repeat(labels, [len(v) for v in prob_vectors])
    df = pd.DataFrame(dict(x=x, g=g))

    if pair_colors is None:
        pal = sns.cubehelix_palette(n_vectors, rot=-.25, light=.7)
    else:
        pal = pair_colors
    
    # The rest of the plotting code comes from the seaborn documentation
    g = sns.FacetGrid(df, row="g", hue="g", aspect=0.5*n_vectors, height=0.5, palette=pal)
    g.map(sns.kdeplot, "x", bw_adjust=.4, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.4)
    g.refline(y=0, linewidth=2, linestyle="-", color='white', clip_on=False)
    
    if centers is not None:
        # insert vertical red lines at the centers
        for center, ax in zip(centers, g.axes.flat):
            ax.axvline(x=center, color='red', linewidth=2, linestyle="-", ymax=0.5, clip_on=False)

        if npmi1 is not None:
            for npmi, ax in zip(npmi1, g.axes.flat):
                ax.axvline(x=npmi, color='lightgrey', linewidth=2, linestyle="-.", ymax=0.5, clip_on=False)

        if npmi2 is not None:
            for npmi, ax in zip(npmi2, g.axes.flat):
                ax.axvline(x=npmi, color='lightgrey', linewidth=2, linestyle=":", ymax=0.5, clip_on=False)

        if npmi3 is not None:
            for npmi, ax in zip(npmi3, g.axes.flat):
                ax.axvline(x=npmi, color='lightgrey', linewidth=2, linestyle="-", ymax=0.5, clip_on=False)

        if pm_ratio is not None:
            for pm, ax in zip(pm_ratio, g.axes.flat):
                ax.axvline(x=pm, color='dodgerblue', linewidth=2, linestyle="--", ymax=0.5, clip_on=False)

    # fix label colors and limits
    for ax in g.axes.flat:
        ax.set_xlim(0, 1)
        ax.xaxis.label.set_color('white')
        ax.xaxis.label.set_size(16)
        ax.tick_params(axis='x', which='major', labelsize=16)
        ax.set_ylim(0, 15)
        for label in ax.get_xticklabels():
            label.set_color('white')
            try:
                label_text = float(label.get_text().replace('−', '-'))
                if label_text < 0 or label_text > 1:
                    label.set_visible(False)
            except ValueError:
                continue

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color='white', fontsize=16,
                ha="left", va="center", transform=ax.transAxes)
    
    g.map(label, "x")
    g.figure.subplots_adjust(hspace=-0.25)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.set_axis_labels(x_var=mode)

    # adjust figure size
    g.fig.set_size_inches(10, n_vectors)
    g.fig.set_dpi(dpi)

    plt.show()
    legend()


def get_pair_colors(data, interesting_pairs, vmax=None, vmin=None):
    #cmap = plt.get_cmap(cmap_name)
    if vmax is None and vmin is None:
        vmax = np.nanmin(data)
        vmin = np.nanmax(data)
    normed_data = 1 - (data - vmin) / (vmax - vmin)
    colors = [mcmap(normed_data[row, col]) for row, col in interesting_pairs]
    return colors


def map_values(vmax, center, vmin):
    # check if center lies between vmax and vmin
    if not vmin <= center <= vmax:
        if vmin > center and vmax > center:
            return 1, 0.5
        elif vmin < center and vmax < center:
            return 0.5, 0
        else:
            raise ValueError("Center lies outside of the range [vmin, vmax]")

    # Calculate the distances
    distance_vc = abs(vmax - center)
    distance_cv = abs(center - vmin)
    
    # Determine which distance is larger
    if distance_vc >= distance_cv:
        larger_distance = distance_vc
    else:
        larger_distance = distance_cv

    scaling_factor = larger_distance / 0.5

    # Map the larger distance to the range [1, 0.5]
    if distance_vc >= distance_cv:
        mapped_vc = 1
        mapped_cv = 0.5 - distance_cv / scaling_factor
    else:
        mapped_cv = 0
        mapped_vc = 0.5 + distance_vc / scaling_factor

    return mapped_vc, mapped_cv



def variability_ratio(matrix, title=None, unit=None, center=1): 
    vmax = np.nanpercentile(matrix, 99)
    vmin = np.nanpercentile(matrix, 1)  
    #assert vmin >= 0

    # shifted ratio center for norm and colormap
    if center == None:
        center = (vmax + vmin) / 2
        print(f"Values centered around: {round(center, 2)}")

    end, start = map_values(vmax, center, vmin)

    # Create a resampled colormap
    cmap = plt.get_cmap('RdBu_r')
    ratio_cmap = cmap(np.linspace(start, end, 256))
    ratio_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', ratio_cmap)

    # style
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8), dpi=dpi)
    plt.title(title)
    plt.imshow(matrix, cmap=ratio_cmap, interpolation='none', vmin=vmin, vmax=vmax)
    plt.colorbar(label=unit)
    plt.xlabel('Locus')
    plt.ylabel('Locus')
            
    plt.grid(False)
    plt.show()


def lineplot_from_matrix(matrices):
    # only keep upper triangle of matrix
    for i, matrix in enumerate(matrices):
        upper_triangle = matrix[np.triu_indices(matrix.shape[0])]
        plt.plot(upper_triangle, label=f'Matrix {i+1}')

    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8), dpi=dpi)
    plt.title(f"Vector")
    plt.xlabel('Locus')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(False)
    plt.show()

