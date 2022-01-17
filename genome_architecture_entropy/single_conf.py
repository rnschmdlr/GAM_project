# %% import
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance

import entropy_measures as em
import toymodel.conf_space as cs

n_slice = 20000

def plot_ensemble(model, mat1, mat2, title, *color):
    # Generate a mask for the upper triangle from value shape, k=1 to see diagonal line in heatmap
    mask_shape = (mat1.shape[1], mat1.shape[1])
    mask = np.triu(np.ones(mask_shape, dtype=bool), k=0)
    cmap = sns.color_palette("flare", as_cmap=True) # Generate a custom diverging colormap
    xmin = np.min(model[0])
    xmax = np.max(model[0])
    ymin = np.min(model[1])
    ymax = np.max(model[1])

    sns.set_theme(style="white")
    fig, (ax1, ax2) = plt.subplots(2, 2)
    fig.set_size_inches(12, 10)
    fig.suptitle(title, size='x-large', weight='bold')

    ax1[0].plot(model[0], model[1], '-k')
    ax1[0].scatter(model[0], model[1], c=color)
    ax1[0].tick_params(labelleft=False, labelbottom=False)
    ax1[0].axis([xmin-0.5, xmax+0.5, ymin-0.5, ymax+0.5])
    ax1[0].set_aspect('equal', 'box')
    ax1[0].set_title('2D chromatin model of loci', loc='left', size='large')
    ax1[0].set(xlabel = 'x', ylabel = 'y')

    ax1[1].set_title('Normalized linkage disequilibirum', loc='left', size='large')
    ax1[1].tick_params(labelleft=False, labelbottom=False)

    ax2[0].set_title('Joint entropy', loc='left', size='large')
    ax2[0].tick_params(labelleft=False, labelbottom=False)
    sns.heatmap(mat1,
                ax=ax2[0],
                mask=mask, 
                cmap=cmap, 
                robust=True, 
                square=True, 
                linewidths=0,
                cbar_kws={"shrink": .82})
    
    ax2[1].set_title('Mutual Information', loc='left', size='large')
    ax2[1].tick_params(labelleft=False, labelbottom=False)
    sns.heatmap(mat2,
                ax=ax2[1],
                mask=mask, 
                cmap=cmap, 
                robust=True, 
                square=True, 
                linewidths=0,
                cbar_kws={"shrink": .82})



# %% test sampling distribution
xy = np.mgrid[0:10:1, 0:10:1].reshape(2,-1)
seg_mat = cs.slice(xy, 0, 9, 0, 9, n_slice)
freq = np.sum(seg_mat, axis=0)
max_freq = np.max(freq)
color = [str(item/max_freq) for item in freq]

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(13, 5)
fig.tight_layout(pad=4, w_pad=0.3)
fig.suptitle('Sampling frequency of loci in slices', size='x-large', weight='bold', x=0.312)
sns.set_theme(style="white")

ax1.scatter(xy[0], xy[1], s=100, c=color)
ax1.tick_params(
        labelleft=False,
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
ax1.axis([0-0.5, 9+0.5, 0-0.5, 9+0.5])
ax1.set_aspect('equal', 'box')
ax1.set_title('Sampling distribution', loc='left', size='large')

ax2.plot(freq / n_slice, '-k')
ax2.axis([-1, xy.size/2, 0, max_freq / n_slice + 0.1])
ax2.set_title('Sampling frequency', loc='left', size='large')
ax2.set(xlabel = 'loci', ylabel = 'frequency')



# %% define model
# model coordinates
new_model_mixed_xy = np.array([ 1.986,   3.770,
                                    1.651,   2.817,
                                    1.781,   1.816,
                                    1.732,   0.807,
                                    0.723,   0.776,
                                    1.421,   0.047,
                                    2.513,   0.534,
                                    2.170,  -0.306,
                                    2.367,  -1.297,
                                    2.612,  -2.276,
                                    2.928,  -3.236,
                                    3.670,  -3.921,
                                    4.680,  -3.882,
                                    5.106,  -2.966,
                                    4.396,  -2.249,
                                    4.446,  -1.240,
                                    3.891,  -0.396,
                                    4.454,   0.442,
                                    5.336,  -0.050,
                                    5.439,  -1.055,
                                    6.255,  -1.650,
                                    7.260,  -1.548,
                                    7.483,  -0.563,
                                    7.147,   0.584,
                                    6.730,  -0.221,
                                    7.861,   0.166,
                                    8.490,   0.957,
                                    8.111,   1.893,
                                    7.620,   2.776,
                                    8.328,   3.496,
                                    9.291,   3.192,
                                    9.734,   2.284,
                                    9.902,   1.288,
                                    9.971,   0.280,
                                    9.834,  -0.720,
                                    9.810,  -1.730])
# reshape pairs to x and y vectors
new_model = np.array([new_model_mixed_xy[0::2], new_model_mixed_xy[1::2]])



# %% entropy of chain
# calculating all to all euclidean distances
dist = distance.cdist(np.stack(new_model, axis=1), np.stack(new_model, axis=1))

# continuos -> discrete by rounding to nearest int 
dist = np.around(dist) # TODO implement Limiting density of discrete points

# calculating shannon-, differential-, joint- entropy and mutual information
print('H =', em.shannon_entropy(dist))
diff_h_chain = em.differential_entropy(dist)
je_dist_mat = em.all_joint_entropy(dist)
mi_dist_mat = em.mutual_information(je_dist_mat)

color = [str(np.abs(item/np.max(diff_h_chain))) for item in diff_h_chain]
plot_ensemble(new_model, je_dist_mat, mi_dist_mat, 'Pairwise distances', color)



# %% joint entropy and mutual information of segregation
# slicing within x, y boundaries
seg_mat = cs.slice(new_model, 0, 10, -5, 5, n_slice)

# calculating shannon-, differential-, joint- entropy and mutual information
print('H =', em.shannon_entropy(seg_mat))
diff_h = em.differential_entropy(seg_mat)
je_mat = em.all_joint_entropy(seg_mat.T)
mi_mat = em.mutual_information(je_mat)

color = [str(np.abs(item/np.max(diff_h))) for item in diff_h]
plot_ensemble(new_model, je_mat, mi_mat, 'Segregation analysis', color)



# %% entropy of lower resolution
# lowering resolution by dropping every second loci
low_res_model = np.array([new_model_mixed_xy[0::4], new_model_mixed_xy[1::4]])

# slicing within x, y boundaries
seg_mat_low = cs.slice(low_res_model, 0, 10, -5, 5, n_slice)

# calculating shannon-, differential-, joint- entropy and mutual information
print('H =', em.shannon_entropy(seg_mat_low))
diff_h_low = em.differential_entropy(seg_mat_low)
je_mat_low = em.all_joint_entropy(seg_mat_low.T)
mi_mat_low = em.mutual_information(je_mat_low)

color = [str(np.abs(item/np.max(diff_h_low))) for item in diff_h_low]
plot_ensemble(low_res_model, je_mat_low, mi_mat_low, 'Segregation analysis (low-res)', color)



# %% plotting differential entropy
# padding and stacking for plotting
diff_h_low_padded = np.insert(diff_h_low, np.arange(0,18,1), None)
values = np.abs(np.stack((diff_h, diff_h_chain, diff_h_low_padded)).T)
data_raw = pd.DataFrame(values, columns=["∆H_orig", "∆H_chain", "∆H_lowres"])
data_raw["∆H_lowres"] = data_raw["∆H_lowres"].interpolate() #fillna(method='pad')
data_normalized = (data_raw - data_raw.min()) / (data_raw.max() - data_raw.min())

fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(12, 5)
sns.lineplot(data=data_normalized, ax=ax1, palette="tab10", linestyle='solid', linewidth=2)
plt.title('Differential Entropy between pairs of loci (abs, normalized)', loc='left')



# %% Series
series = np.load('toymodel.npy')
for state in range(series.shape[0]):
    model = series[state].T

    # slicing within x, y boundaries
    seg_mat = cs.slice(model, 0, 10, -5, 5, n_slice)

    # calculating shannon-, differential-, joint- entropy and mutual information
    print('H =', em.shannon_entropy(seg_mat))
    diff_h = em.differential_entropy(seg_mat)
    je_mat = em.all_joint_entropy(seg_mat.T)
    mi_mat = em.mutual_information(je_mat)

    title = 'Segregation analysis T=%d' % (state+1)
    plot_ensemble(model, je_mat, mi_mat, title)
