# %% Imports and helper function definitons
"""Imports and helper function definitons"""
import os
os.chdir('/Users/pita/Documents/Rene/GAM_project/genome_architecture_entropy/src/')


import numpy as np
import pandas as pd; pd.set_option("display.precision", 3)
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import multiprocessing ;multiprocessing.set_start_method('fork', force=True)

import entropy.transfer_entropy.compute_transfer_entropy as em
import entropy.transfer_entropy.compute_direct_transfer_entropy as dte
import toymodel.sampling
import entropy.transfer_entropy.plotting as plotting
import entropy.shannon_entropies.compute_2d_entropies as e2d


def shuffle_with_fixed_element(array, axis, index):
    assert axis < array.ndim, "Axis is out of bounds for array"

    # Get list of indices excluding the fixed one
    idx = np.delete(np.arange(array.shape[axis]), index)

    # Shuffle the indices
    np.random.shuffle(idx)

     # Insert the fixed index back to its original place
    idx = np.insert(idx, index, index)

    # Create a tuple for advanced indexing
    idx_tuple = [slice(None)]*axis + [idx]

    # Shuffle the array along idx
    result = array[tuple(idx_tuple)]
    
    return result


def top_changing_loci(array, frm_n, top_n):
    # compute cummulative differences in net influence from n to end and select highest
    df = pd.DataFrame(array)
    #df.columns += 1

    diff = df.diff(axis=0).abs()
    # scale difference to summed influence 
    diff = diff / df.sum(axis=0)
    diff = diff[frm_n-1:].sum(axis=0).sort_values(ascending=False)
    return diff.index[:top_n]


def symmetry_analysis(mat1, mat2, rows):
    # Compute the difference matrix
    diff_matrix = mat1 - mat2

    # Separate positive and negative differences
    positive_diff = diff_matrix.copy()
    positive_diff[positive_diff < 0] = 0

    negative_diff = diff_matrix.copy()
    negative_diff[negative_diff > 0] = 0

    # Compute total positive and negative differences
    total_positive_diff = positive_diff.sum()
    total_negative_diff = negative_diff.sum()
    print('tot_pos', round(total_positive_diff,2), 'tot_neg', round(total_negative_diff,2))

    # Compute the differences for specific rows
    specific_diff_matrix = mat1[rows] - mat2[rows]

    # Separate positive and negative differences for specific rows
    specific_positive_diff = specific_diff_matrix[specific_diff_matrix > 0].sum()
    specific_negative_diff = specific_diff_matrix[specific_diff_matrix < 0].sum()
    print('pos_abs', round(specific_positive_diff,2), 'neg_abs', round(specific_negative_diff,2))

    # Compute the percentages
    positive_percentage = (specific_positive_diff / total_positive_diff) * 100 if total_positive_diff != 0 else 0
    negative_percentage = (specific_negative_diff / total_negative_diff) * 100 if total_negative_diff != 0 else 0

    return positive_percentage, negative_percentage


def unique_vectors(arr):
    min_unique = np.inf
    unique_rows_per_n = []
    
    for n in range(arr.shape[0]):
        subarr = arr[n]
        unique_rows = np.unique(subarr, axis=0)
        unique_rows_per_n.append(unique_rows)
        if unique_rows.shape[0] < min_unique:
            min_unique = unique_rows.shape[0]
            
    new_arr = np.empty((arr.shape[0], min_unique, arr.shape[2]))

    for n in range(arr.shape[0]):
        new_arr[n] = unique_rows_per_n[n][:min_unique]
        
    return new_arr


# %%   1. Load model
'''1. Load model'''
path_model = '../data/toymodels/model4/4_3/'
model = 'toymodel4_3_multi_500'
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



# %%    - Quick coordinate viewer
'''- Quick coordinate viewer'''
plt.rcParams["figure.figsize"] = (6,2)
plt.rcParams['figure.dpi'] = 300
# colormap from grey to black in 9 steps
cmap = plt.cm.get_cmap('Greys', 12)
import cmcrameri.cm as cmc
cmap = cmc.grayC_r(np.linspace(0.2, 1, 9))

# hex code color list from grey to black in 9 steps
colors = ['#e6e6e6', '#d4d4d4', '#c2c2c2', '#b0b0b0', '#9e9e9e', '#8c8c8c', '#7a7a7a', '#565656', '#000000']
# '#e6e6e6', '#d4d4d4',

for realisation in range(min(5, n_realisations)):
    for state in range(n_timesteps):
        plt.plot(series[realisation].T[0,:,state], series[realisation].T[1,:,state], linewidth=2, color=colors[state])
        plt.tick_params(labelleft=False, labelbottom=False)
    plt.savefig(path_model + model + '_realisation' + str(realisation) + '.png', bbox_inches='tight')
    #plt.show()
    plt.clf()



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



# %%   3. Compute segregation matrix
'''3. Compute segregation matrix'''
n_states = n_realisations * n_timesteps
n_slice = 200
wdf_lb_ub = [0.05, 1]
seg_mats = np.empty((n_states, n_slice, series_.shape[1]))

for state in tqdm(range(n_states)):
    coords_model = series_[state].T
    seg_mats[state] = toymodel.sampling.slice(coords_model, n_slice, wdf_lb_ub)

print('>> Segregation matrix computed with', n_slice, 'slices,', wdf_lb_ub[0], 'lower bound and', wdf_lb_ub[1], 'upper bound window detection frequency')

if n_realisations > 1:
    seg_mats = np.reshape(seg_mats, newshape=(n_timesteps, n_realisations * n_slice, n_loci), order='C') # only for grouped!
    print('>> Realisations have been grouped for', n_realisations * n_slice, 'for each timestep')

model_seg = '_seg_mats_%d_t%d' % (n_realisations * n_slice, n_timesteps)
np.save(path_model + model + model_seg + '.npy', seg_mats)



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



# %%    - Window detection statistics
'''- Window detection statistics'''
wdf_lb_ub = [0.11, 1]
n_slice_ = n_realisations * n_slice
detect_sum = np.arange(int(n_loci*wdf_lb_ub[0]), int(n_loci*wdf_lb_ub[1])+1)
n_detect = np.zeros(shape=(n_timesteps, len(detect_sum)))
n_unique = []
wdf_mean = []
wdf_min = []
wdf_max = []

print('Timesteps:', n_timesteps, '\nRealisations:', n_realisations, '\nSlices:', n_slice, '\nLoci:', n_loci)
for t in range(n_timesteps):
    wdf = np.sum(seg_mats[t], axis=1) / n_loci
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

idx = np.arange(n_timesteps+n_cut_pre)[n_cut_pre:].tolist()
df = pd.DataFrame(d, index=idx)
df2 = pd.DataFrame(n_detect, columns=detect_sum, index=idx)
df_ = pd.concat([df, df2], axis=1)

file = path_model + model + model_seg + '_wdf_stats.csv'
df_.to_csv(file, sep=';')
df_



# %%    - Shuffle time order
'''- Shuffle time order'''
seq = [5,3,0,6,1,2,7,4]
seg_mats_ = seg_mats[np.array(seq),:,:]; print('Sequence = ', seq)

# backwards time order
#seg_mats_ = np.flip(seg_mats)

# random time order
#seg_mats_ = np.random.RandomState().permutation(seg_mats)



# %%   4. Pairwise TE calculation
'''4. Pairwise TE calculation'''
hist_len = 1
TE = em.pw_transfer_entropy(seg_mats, hist_len)
TA = 0.5 * (TE - TE.T)

seg_mats_stacked = np.reshape(seg_mats, newshape=(n_timesteps * n_realisations  * n_slice, n_loci))
npmi_mat_stacked = e2d.npmi_2d_fast(seg_mats_stacked.T.astype(int), seg_mats_stacked.T.astype(int))

res = {'Permutation': [np.arange(n_timesteps)],
       'Kendalls tau-b (perm., seq.)': 1,
       #'Net direction': np.sum(np.tril(TE) - np.triu(TE)),
       'Median TE': np.median(TE),
       'Mean TE': np.mean(TE),
       'Standard deviation': np.std(TE),
       'Total sum TE': np.sum(TE),
       'Total asymmetrical TE sum ': np.sum(TA[TA > 0]),}

file_heatmaps = path_model + model + model_seg + '_te_matrices2.png'
file_cbar = path_model + model + model_seg + 'cbar'
plotting.heatmaps(TE, np.max(TE), np.max(TA), TA, npmi_mat_stacked)
#plotting.generate_colorbar('Transfer entropy (bits)', [0, 0.12, 0.25, 0.37, 0.5], ['0.0', 0.12, 0.25, 0.37, '0.50'], 'v', 'left', (0.3, 5), 'blue', file_cbar+'te.png')
#plotting.generate_colorbar('Net transfer entropy (bits)', [-0.15, -0.14, -0.07, 0, 0.07, 0.14, 0.15], ['', -0.14, -0.07, '0.0', 0.07, 0.14, ''], 'v', 'right', (0.3, 5), 'green-blue', file_cbar+'te_asym.png')
#plotting.generate_colorbar('NPMI', [-1, -0.5, 0, 0.5, 1], ['-1.0', -0.5, '0.0', 0.5, '1.0'], 'h', 'bottom', (5, 0.3), 'green-blue', file_cbar+'npmi.png')

pd.DataFrame(res)



# %%    - Signifance testing pairwise TE
'''- Signifance testing pairwise TE'''
rep = 1000
temp = np.empty(shape=(rep, n_loci, n_loci))
bool_arr = np.zeros_like(temp)
seg_mats_ = seg_mats.copy()

for itr in tqdm(range(rep)):
    temp[itr] = em.pw_transfer_entropy(seg_mats, hist_len, stattest=True)
    bool_arr[itr] = np.greater_equal(np.abs(temp[itr]), np.abs(TE))

props = bool_arr.sum(axis=0) / rep
props = props - np.diag(np.diag(props))

# Create a new array for categorizing p-values
pval = np.zeros_like(props)

# Assign different integer values based on p-value ranges
pval[props >= 0.05] = 1
pval[(props < 0.05) & (props >= 0.01)] = 2
pval[(props < 0.01) & (props >= 0.001)] = 3
pval[props < 0.001] = 4
pval_df = pd.DataFrame(pval)

file_significance = path_model + model + model_seg + '_te_significance_driver.png'
plotting.te_significance(pval_df, file_significance)



# %%    - Significance testing driver identification
'''- Significance testing driver identification'''
rep = 300
bool_driver = np.empty(shape=(n_loci, rep))
bool_bystander = np.empty(shape=(n_loci, rep))

# observed net influence
tap = TA.copy()
tap[tap < 0] = 0
driver = np.sum(tap, axis=1)
bystander = np.sum(tap, axis=0)
v = (driver - bystander) 

for locus in tqdm(range(n_loci)):
    for itr in range(rep):
        seg_mats_ = shuffle_with_fixed_element(seg_mats.copy(), 2, locus)
        #seg_mats_ = np.moveaxis(seg_matscopy(), 2, 0)
        #seg_mats_p = np.random.RandomState().permutation(seg_mats_)
        #seg_mats_ = np.moveaxis(seg_mats_p, 0, 2)

        # construct null distribution v_p
        temp_te = em.pw_transfer_entropy(seg_mats_, hist_len)
        temp_ta = 0.5*(temp_te - temp_te.T)
        temp_ta[temp_ta < 0] = 0
        driver = np.sum(temp_ta, axis=1)[locus] #rowsum
        bystander = np.sum(temp_ta, axis=0)[locus] #columnsum
        v_p = (driver - bystander)

        v_p = np.greater(v_p, 0)
        v[locus] = np.greater(v[locus], 0)

        # if both true -> 1, else 0
        bool_driver[locus, itr] = np.logical_and(v_p, v[locus])

        # if both false -> 1, else 0
        bool_bystander[locus, itr] = np.logical_not(np.logical_or(v_p, v[locus]))

        # compare v to v_p to empirically
        #bool_arr[locus, itr] = np.greater_equal(v_p, v[locus])

# calculate p-values, discard diagonal
props_driver = 1 - bool_driver.sum(axis=1) / rep
print(np.where(props_driver < 0.05)[0]+1)

props_bystander = 1 - bool_bystander.sum(axis=1) / rep
print(np.where(props_bystander < 0.05)[0]+1)
#significant = np.where(props_driver < 0.05)[0]
#drivers = np.where(v > 0)[0]
#significant_drivers = np.intersect1d(significant, drivers)

#props = props - np.diag(np.diag(props))

# Create a new array for categorizing p-values
#pval = np.zeros_like(props)
#pval = np.diag(props)
## Assign different integer values based on p-value ranges
#pval[props >= 0.05] = 1
#pval[(props < 0.05) & (props >= 0.01)] = 2
#pval[(props < 0.01) & (props >= 0.001)] = 3
#pval[props < 0.001] = 4
#pval_df = pd.DataFrame(pval)
#
#file_significance = path_model + model + model_seg + '_te_significance_driver.png'
#plotting.te_significance(pval_df, file_significance)


# %%   5. Plot progression and TE matrices
'''5. Plot progression and TE matrices'''
# precompute max values for scaling
repr = 3
te_mat = np.zeros(shape=(n_timesteps, n_loci, n_loci))
net_influence_array = np.zeros(shape=(n_timesteps, n_loci))
driver_array = np.zeros(shape=(n_timesteps, n_loci))
bystander_array = np.zeros(shape=(n_timesteps, n_loci))
te_vmax = 0
ta_vmax = 0
ta_vmin = 0
vmax_interact = 0
vmax = 0
vmin = 0

start = hist_len + n_cut_pre
for n in range(start, n_timesteps):
    seg_mat = seg_mats[:n+start] 
    te_mat[n] = em.pw_transfer_entropy(seg_mat, hist_len)
    te_vmax = max(te_vmax, np.max(te_mat[n]))

    # positive asym TE
    te_mat_asym = 0.5 * (te_mat[n]- te_mat[n].T)
    tap = te_mat_asym.copy()
    tap[tap < 0] = 0
    ta_vmax = max(ta_vmax, np.max(te_mat_asym))
    ta_vmin = -ta_vmax

    # interaction strength
    driver = np.sum(tap, axis=1)
    bystander = np.sum(tap, axis=0)
    vmax_interact = max(vmax_interact, max(driver - bystander))


# compute for each timestep:
for n in range(0, n_timesteps):
    coords_model = series[repr, n].T

    if n >= start:
        te_mat_asym = 0.5 * (te_mat[n] - te_mat[n].T)
        print('Total asym TE:', np.sum(np.abs(te_mat_asym)))
        print('Total TE:', np.sum(np.abs(te_mat[n])))
        tap = te_mat_asym.copy()
        tap[tap < 0] = 0
        driver = np.sum(tap, axis=1)
        bystander = np.sum(tap, axis=0)
        net_influence = (driver - bystander) / vmax_interact
        driver_array[n] = driver / vmax_interact
        bystander_array[n] = bystander / vmax_interact
        net_influence_array[n] = net_influence
    else:
        net_influence = np.zeros(n_loci)
        te_mat_asym = np.zeros((n_loci, n_loci))

    # prepare plot function calls
    file1 = path_model + model + model_seg + '_progression_coord_bw%03d.png' % (n)
    plotting.coords(coords_model, net_influence, file1)

for n in range(start, n_timesteps):
    file2 = path_model + model + model_seg + '_progression_matrices%03d.png' % (n)
    plotting.heatmaps(te_mat[n], te_vmax, ta_vmax, te_mat_asym, file2)

#plotting.generate_colorbar('Net influence % of max', [-1, -0.5, 0, 0.5, 1], [-1.0, -0.5, 0.0, 0.5, 1.0], 'v', 'right', (0.3, 3), 'green-blue', file_cbar+'net_influence_v.png')
#plotting.generate_colorbar('Net influence % of max', [-1, -0.5, 0, 0.5, 1], [-1.0, -0.5, 0.0, 0.5, 1.0], 'h', 'bottom', (5, 0.3),'green-blue', file_cbar+'net_influence_h.png')



# %%    - TE matrices differences
'''5.1 TE matrices differences'''
# calculate differences between matrices
te_mat_diff = np.zeros(shape=(n_timesteps, n_loci, n_loci))
for n in range(start, n_timesteps):
    te_mat_diff[n] = te_mat[n] - te_mat[n-1]

# plot differences
for n in range(0, n_timesteps):
    print(n)
    file3 = path_model + model + model_seg + '_progression_matrices_diff%03d.png' % (n)
    plotting.heatmaps(te_mat[n], te_vmax, te_vmax/10, te_mat_diff[n], file3)



# %%    - Loci trajectories
'''5.2 Loci trajectories'''
index = pd.MultiIndex.from_product(
    [np.arange(n_timesteps), np.arange(n_loci)], 
    names=['Timepoint', 'Locus'])

data = {
    'Net': net_influence_array.flatten(),
    'Positive': driver_array.flatten(),
    'Negative': -bystander_array.flatten()
}
influence_df = pd.DataFrame(data, index=index).drop(0, level='Timepoint')
influence_df.index = influence_df.index.set_levels(influence_df.index.get_level_values(1)[:n_loci] + 1, level=1)  # Increment 'locus' index
influence_df_melt = influence_df.reset_index().melt(id_vars=['Timepoint', 'Locus'], var_name='Influence', value_name='Influence % of max')

# explorative analysis: top changing loci influences
plotting.trajectories(influence_df_melt, top_changing_loci(net_influence_array, 3, 3))
plotting.trajectories(influence_df_melt, top_changing_loci(driver_array, 3, 3))
plotting.trajectories(influence_df_melt, top_changing_loci(-bystander_array, 3, 3))

# specific loci
loci = [15,54] # left subTAD boundary
loci = [65,85] # right subTAD boundary
loci = [10,90] # TAD boundary
file_trajectories = path_model + model + model_seg
plotting.trajectories(influence_df_melt, loci, file_trajectories + f'_trajectories_{loci}.png')

# trajectories statistics
# difference in negative influence between timepoints 4 and 8 for loci 19-23 in percent
influence_df.loc[8].loc[19:23, 'Negative'].sum()/5 - influence_df.loc[4].loc[19:23, 'Negative'].sum()/5 * 100

# sum of positive influence of all loci at timepoint 8
influence_df.loc[8].loc[:, 'Positive'].sum()



# %%    - Symmetry analysis
loci1 = [2,3,4,5]
loci2 = [0,1]
loci3 = [5]
for i, n in enumerate ([3,6]):
    print('\n\nn =', n)
    te_mat_asym = 0.5 * (te_mat[n]  - te_mat[n] .T)
    ta_left = te_mat_asym[:50]
    ta_right = np.flipud(te_mat_asym[50:])

    ta_pos = te_mat_asym[te_mat_asym > 0].sum()
    ta_left_pos = ta_left[ta_left > 0].sum()
    ta_right_pos = ta_right[ta_right > 0].sum()
    pos = (ta_left_pos - ta_right_pos) / ta_pos * 100
    print('Positive influence')
    print('left vs right side loop:', round((ta_left_pos - ta_right_pos), 2))
    print('left vs right side loop:', round(pos, 2), '% of total pos TA')

    ta_neg = np.abs(te_mat_asym[te_mat_asym < 0].sum())
    ta_left_neg = ta_left[ta_left < 0].sum()
    ta_right_neg = ta_right[ta_right < 0].sum()
    neg = (ta_left_neg - ta_right_neg) / ta_neg * 100
    print('Negeative influence')
    print('left vs right side loop:', round((ta_left_neg - ta_right_neg), 2))
    print('left vs right side loop:', round(neg, 2), '% of total neg TA', '\n',)

   #pos, neg = symmetry_analysis(ta_left, ta_right, loci1)
   #print(loci1, 'vs right side loop:', round(pos, 2), '% of total pos TA')
   #print(loci1, 'vs right side loop:', round(neg, 2), '% of total neg TA', '\n',)

   #pos, neg = symmetry_analysis(ta_left, ta_right, loci2)
   #print(loci2, 'vs left side loop:', round(pos, 2), '% of total pos TA')
   #print(loci2, 'vs left side loop:', round(neg, 2), '% of total neg TA', '\n',)

   #pos, neg = symmetry_analysis(ta_right, ta_left, loci3)
   #print(loci3, 'vs left side loop:', round(pos, 2), '% of total pos TA')
   #print(loci3, 'vs left side loop:', round(neg, 2), '% of total neg TA')




# %%    - Plot network
'''- Plot network'''
n = 5
coords_model = series[repr, n].T
te_mat_asym = 0.5 * (te_mat[n] - te_mat[n].T)
te_asym = te_mat_asym.copy()

'''
# key
te_asym[:, :14] = 0 # no influence to zeroed out nodes
te_asym[:14, :] = 0 # no influence from zeroed out nodes
# if both, completely disregard nodes

te_asym[:14, :14] = 0 # no internal influences
te_asym[14:, 14:] = 0 # only cross influences

te_asym[14:, :] = 0
te_asym[:, 14:] = 0
# if both only internal influences
'''

#for model 2
## left loop all outgoing
#te_asym[14:, :] = 0
## left loop bystander
#te_asym[10:, :] = 0
## right loop all outgoing
#te_asym[:14, :] = 0
## right loop, incoming influences, excluding 26,27,28
#te_asym[:, :14] = 0
#te_asym[:, 25:] = 0
#
## right loop incoming to 20:23
#te_asym[:,:19] = 0
#te_asym[:,23:] = 0


# for model 3
#te_asym[:, :12] = 0
#te_asym[:, 90:] = 0
#
#te_asym[:50, :] = 0
#te_asym[:, :50] = 0

file_network = path_model + model + model_seg + '_te_network_t%d_2.png' % n
plotting.network(te_asym.copy(), coords_model, list(net_influence_array[n]), file_network)



# %%    - Plot net influence distribution
'''- Plot net influence distribution'''
for n in range(7, n_timesteps):
    asym_influence_df = pd.DataFrame({'T_A positive': driver_array[n], 
                                    'T_A negative': -bystander_array[n], 
                                    'T_A net': net_influence_array[n]})
    asym_influence_df.index += 1
    asym_influence_df['Locus'] = asym_influence_df.index

    file_net_influence = path_model + model + model_seg + '_te_net_influence_step_%d.png' % n
    plotting.net_influence_bars(asym_influence_df)


# filter out loci from asym_influence_df with negative T_A net influence
filter_df = asym_influence_df[asym_influence_df['T_A net'] > 0]

# compute ratio of positive+negative to net influence
np.mean(((filter_df['T_A positive'] - filter_df['T_A net']) / filter_df['T_A positive'])[:19])






