# %% import
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, colors
from scipy.spatial import distance
from scipy import stats

#import pyximport; pyximport.install()
import cosegregation_internal as ci
import entropy_measures as em
import toymodel.conf_space as cs

n_slice = 100000

def plot_ensemble(model, mat1, mat2, cosegregation, title, entropy, color_points):
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
    fig.suptitle(title, fontsize=20, weight='bold')
    plt.figtext(0.12, 0, 'Overall entropy \nH = '+entropy, fontsize=16, va="top", ha="left")

    ax1[0].plot(model[0], model[1], '-k') #, c=color_line)
    ax1[0].scatter(model[0], model[1], s=125, c=color_points)
    ax1[0].tick_params(labelleft=False, labelbottom=False)
    ax1[0].axis([xmin-0.5, xmax+0.5, ymin-0.5, ymax+0.5])
    ax1[0].set_aspect('equal', 'box')
    ax1[0].set_title('2D chromatin model of loci', loc='left', fontsize=16)
    ax1[0].set(xlabel = 'x', ylabel = 'y')

    ax1[1].set_title('Mutual Information', loc='left', fontsize=16)
    ax1[1].tick_params(labelleft=False, labelbottom=False)
    sns.heatmap(mat2,
                ax=ax1[1],
                mask=mask,
                cmap=cmap,
                robust=True,
                square=True,
                linewidths=0,
                cbar_kws={"shrink": .82})

    ax2[0].set_title('Normalized linkage disequilibirum', loc='left', fontsize=16)
    ax2[0].tick_params(labelleft=False, labelbottom=False)
    sns.heatmap(cosegregation,
                ax=ax2[0],
                mask=mask, 
                cmap=cmap, 
                robust=True, 
                square=True, 
                linewidths=0,
                cbar_kws={"shrink": .82})
    
    ax2[1].set_title('Joint Entropy', loc='left', fontsize=16)
    ax2[1].tick_params(labelleft=False, labelbottom=False)
    sns.heatmap(mat1,
                ax=ax2[1],
                mask=mask, 
                cmap=cmap, 
                robust=True, 
                square=True, 
                linewidths=0,
                cbar_kws={"shrink": .82})



# %% test sampling distribution
n_slice = 100000

xy = np.mgrid[0:10:1, 0:10:1].reshape(2,-1)
seg_mat = cs.slice(xy, n_slice)
freq = np.sum(seg_mat, axis=0)
rel_freq = freq / n_slice
mean = np.round(np.mean(rel_freq), 4)
std = np.round(np.std(rel_freq), 4)
max_freq = np.max(freq)

color = [1 - item/max_freq for item in freq]
cmap = plt.cm.gray
norm = colors.Normalize(vmin=0.0, vmax=1.0)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(13, 5)
fig.tight_layout(pad=4, w_pad=0.3)
fig.suptitle('Probability of loci in slices', size='x-large', weight='bold')
sns.set_theme(style="white")

ax1.scatter(xy[0], xy[1], s=100, c=cmap(norm(color)))
ax1.tick_params(
        labelleft=False,
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
ax1.axis([0-0.5, 9+0.5, 0-0.5, 9+0.5])
ax1.set_aspect('equal', 'box')
ax1.set_title('Empirical probability', loc='left', size='large')

ax2.plot(rel_freq, '-k')
ax2.axis([-1, xy.size/2, rel_freq.min() - 0.3 * mean, rel_freq.max() + 0.3 * mean])
ax2.set_title('Relative frequency', loc='left', size='large')
ax2.set(xlabel = 'loci', ylabel = 'frequency')

plt.axhline(mean, color='r')
string = 'mean = ' + str(mean*100) + '%; SD = ' + str(std*100) + '%; ' + str(np.around(std/mean*100, 2)) + ' % mean'
plt.figtext(0.14, 0, string, fontsize=12, va="top", ha="left")




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
entropy = str(np.around(em.shannon_entropy(dist), 2))
win_h_chain = em.window_entropy(dist)
diff_h_chain = em.differential_entropy(dist)
je_dist_mat = em.all_joint_entropy(dist)
mi_dist_mat = em.mutual_information(je_dist_mat)

color_line = [str(np.abs(item/np.max(diff_h_chain))) for item in diff_h_chain]
# color_points = [str(1-3*(1-(item/np.max(win_h_chain)))) for item in win_h_chain]
color_points = [str(1-np.abs(item/np.max(mi_dist_mat))) for item in mi_dist_mat[0]]
plot_ensemble(new_model, 
            je_dist_mat, 
            mi_dist_mat, 
            np.empty(shape=(n_slice, 36)), #TODO
            'Pairwise distances', 
            entropy,  
            color_points)



# %% joint entropy and mutual information of segregation
# slicing within x, y boundaries
n_slice = 100000
seg_mat = cs.slice(new_model, n_slice)

# calculate normalized linkage disequilibrium
cosegregation_raw = ci.dprime_2d(seg_mat.T.astype(int), seg_mat.T.astype(int))
cosegregation = cosegregation_raw[cosegregation_raw < 0] = 0

# calculating shannon-, differential-, joint- entropy and mutual information
entropy = str(np.around(em.shannon_entropy(seg_mat), 2))
diff_h = em.differential_entropy(seg_mat)
je_mat = em.all_joint_entropy(seg_mat.T)
mi_mat = em.mutual_information(je_mat)

mi_mat = mi_mat - np.diag(np.diag(mi_mat))
mi_sum = np.sum(mi_mat, axis=0)
color_points = [1 - np.abs(item / np.max(mi_sum)) for item in mi_sum]

plot_ensemble(new_model, 
            je_mat, 
            mi_mat, 
            cosegregation,
            'Segregation analysis', 
            entropy, 
            color_points)

# diagonal has to be removed to norm the values of interest
cosegregation = cosegregation - np.diag(np.diag(cosegregation))
cosegregation = cosegregation - cosegregation.min()
cosegregation = cosegregation / cosegregation.max()
mi_mat = mi_mat - mi_mat.min()
mi_mat = mi_mat / mi_mat.max()

diff = (cosegregation - mi_mat) / cosegregation

cosegregation_raw = cosegregation_raw - np.diag(np.diag(cosegregation_raw))
cosegregation_raw = cosegregation_raw - cosegregation_raw.min()
cosegregation_raw = cosegregation_raw / cosegregation_raw.max()
cosegregation_sum = np.sum(cosegregation_raw, axis=0)
color_points = [1 - np.abs(item / np.max(cosegregation_sum)) for item in cosegregation_sum]

plot_ensemble(new_model, 
            diff, 
            mi_mat, 
            cosegregation,
            'Segregation comparison', 
            entropy, 
            color_points)



# %% entropy of lower resolution
# lowering resolution by dropping every second loci
low_res_model = np.array([new_model_mixed_xy[0::4], new_model_mixed_xy[1::4]])

# slicing within x, y boundaries
seg_mat_low = cs.slice(low_res_model, 0, 10, -5, 5, n_slice)

# calculate normalized linkage disequilibrium
cosegregation = ci.dprime_2d(seg_mat_low)

# calculating shannon-, differential-, joint- entropy and mutual information
entropy = str(np.around(em.shannon_entropy(seg_mat_low), 2))
diff_h_low = em.differential_entropy(seg_mat_low)
je_mat_low = em.all_joint_entropy(seg_mat_low.T)
mi_mat_low = em.mutual_information(je_mat_low)
h_max = np.max([je_mat, mi_mat])

color_line = [str(np.abs(item/np.max(diff_h_low))) for item in diff_h_low]
color_points = [str(1-np.abs(item/np.max(mi_mat_low))) for item in mi_mat_low[0]]
plot_ensemble(low_res_model, 
            je_mat_low, 
            mi_mat_low, 
            'Segregation analysis (low-res)', 
            entropy, 
            color_points)



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
