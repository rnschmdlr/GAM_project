# %% Imports and helper function definitons
"""Imports and helper function definitons"""
import os
os.chdir('/Users/pita/Documents/Rene/GAM_project/genome_architecture_entropy/src/')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.auto import tqdm

import variability_analysis.estimate_variability as vrb
import variability_analysis.plotting as plot
import variability_analysis.statistic_functions as statfx
import entropy.shannon_entropies.compute_2d_entropies as e2d



def select_region(data, chr, xmb, ymb, size, res_kb=50):
    seg_table_chr = data[data['chrom'] == chr]
    seg_table_chr = seg_table_chr.drop(columns=['chrom', 'start', 'stop'])
    seg_mat = np.array(seg_table_chr, dtype=bool)

    # base region to matrix coordinates
    x = int(xmb * 1000 / res_kb)
    y = int(ymb * 1000 / res_kb)
    s = int(size * 1000/ res_kb)

    # conversion to slices
    start = 0 + min(x, y)
    end = start + s
    seg_mat_region = seg_mat[start:end, :]
    
    return seg_mat_region


def npmi_wrapper(matrix):
    matrix = e2d.npmi_2d_fast(matrix.T.astype(int), matrix.T.astype(int))
    return matrix


def get_prob_vectors(loci_pairs_to_extract, prob_dict):
    """Get specific probability vectors based on loci pair numbers from a dictionary."""
    return [prob_dict[pair[0]][pair[1]] for pair in loci_pairs_to_extract]


def select_interesting_pairs(data, num_each=3):
    # Boolean mask for upper triangle
    mask = np.triu(np.ones(data.shape, dtype=bool), k=1)

    # Mask for diagonal
    mask[np.diag_indices(data.shape[0])] = False
    # Mask for nans
    mask[np.isnan(data)] = False

    # Flatten data and get sorted indices
    flat_data = data[mask]
    sorted_indices = np.argsort(flat_data)

    # Get highest and lowest scoring pairs
    highest_indices = sorted_indices[-num_each:]
    highest_indices = highest_indices[::-1]
    #lowest_indices = sorted_indices[:num_each]
    #lowest_indices = lowest_indices[::-1]

    # Get pairs around mean and median
    #mean_val = np.mean(flat_data)
    median_val = np.median(flat_data)
    #mean_indices = np.argsort(np.abs(flat_data - mean_val))[:num_each]
    median_indices = np.argsort(np.abs(flat_data - median_val))[:num_each]

    # Combine all the indices
    interesting_indices = np.concatenate([highest_indices, median_indices])

    # Create a mapping from 1D indices to original 2D indices
    row_indices, col_indices = np.where(mask)
    interesting_row_indices = row_indices[interesting_indices]
    interesting_col_indices = col_indices[interesting_indices]

    interesting_pairs = list(zip(interesting_row_indices, interesting_col_indices))

    return interesting_pairs



# %% Example
matrix = np.array([[0, 1, 0, 1],
                   [1, 1, 0, 1],
                   [0, 1, 0, 0],
                   [1, 0, 1, 0],
                   [0, 0, 1, 1]])
pair = (1, 3)

#print(calculate_probability(matrix, pair))
all_pair_probabilities = vrb.calculate_probs_mat(matrix)

# For each pair, print the probabilities
for pair, probabilities in all_pair_probabilities.items():
    print(f"For pair {pair}:")
    print(probabilities)
    print("------")



# %% Load Models
path_model = '../data/my_models/model8/'
modelA = 'modelA_multi_100'
seriesA = np.load(path_model + modelA + '.npy')
seg_matsA = np.load(path_model + 'modelA_seg_mats_1000_t1.npy')
seg_mats_A = np.reshape(seg_matsA, newshape=(1 * 1000, 36), order='F').T

modelB = 'modelB_multi_100'
seriesB = np.load(path_model + modelB + '.npy')
seg_matsB = np.load(path_model + 'modelB_seg_mats_1000_t1.npy')
seg_mats_B = np.reshape(seg_matsB, newshape=(1 * 1000, 36), order='F').T

modelC = 'modelC_multi_100'
seriesC = np.load(path_model + modelC + '.npy')
seg_matsC = np.load(path_model + 'modelC_seg_mats_1000_t1.npy')
seg_mats_C = np.reshape(seg_matsC, newshape=(1 * 1000, 36), order='F').T

modelD = 'modelD_multi_100'
seg_matsD = np.load(path_model + 'modelD_seg_mats_1000_t1.npy')
seg_mats_D = np.reshape(seg_matsD, newshape=(1 * 1000, 36), order='F').T

plot.contact_map(npmi_wrapper(seg_mats_A))
plot.contact_map(npmi_wrapper(seg_mats_B))
plot.contact_map(npmi_wrapper(seg_mats_C))
plot.contact_map(npmi_wrapper(seg_mats_D))



# %% Load experimental dataset
"""Load experimental dataset"""""
path = '/Users/pita/Documents/Rene/GAM_project/genome_architecture_entropy/data/experimental/'

#data_unphased = path + 'F123.all.as.3NPs.mm10.curated.segregation_at50000.passed_qc_fc5_cw6_s11.table'
#seg_table_unphased = pd.read_table(data_unphased)

data_paternal = path + 'F123.All.as.3NPs.mm10.curated.CAST.segregation_at50000.passed_qc_fc5_cw6_s11.table'
seg_table_paternal = pd.read_table(data_paternal)

data_maternal = path + 'F123.All.as.3NPs.mm10.curated.S129.segregation_at50000.passed_qc_fc5_cw6_s11.table'
seg_table_maternal = pd.read_table(data_maternal)



# %% Specify region and plot npmi contact map
"""Specify region and plot npmi contact map"""
## 13, 78, 78, 7
## 2, 92, 92, 4
## 3, 88, 88, 4
## 5, 52, 52, 4
# 1, 110.6, 110.6, 3
# 5, 20, 20 , 8
chrom = 'chrX'
x,y = 1,1
s = 2
print(chrom, x,y, s)
#seg_mat_unphased = select_region(seg_table_unphased, 'chr13', x, y, s)
#npmi_unphased = e2d.npmi_2d_fast(seg_mat_unphased.astype(int), seg_mat_unphased.astype(int))
#np.fill_diagonal(npmi_unphased, np.nan)
#plot.contact_map(npmi_unphased, 'U')

seg_mat_paternal = select_region(seg_table_paternal, chrom, x, y, s)
npmi_paternal = e2d.npmi_2d_fast(seg_mat_paternal.astype(int), seg_mat_paternal.astype(int))
#np.fill_diagonal(npmi_paternal, np.nan)
#plot.contact_map(npmi_paternal, 'P')

seg_mat_maternal = select_region(seg_table_maternal, chrom, x, y, s)
npmi_maternal = e2d.npmi_2d_fast(seg_mat_maternal.astype(int), seg_mat_maternal.astype(int))
#np.fill_diagonal(npmi_maternal, np.nan)
#plot.contact_map(npmi_maternal, 'M')

# unload data
#del data_paternal, data_maternal


# combine seg_table_maternal and seg_table_paternal 
# merge the two matrixes by intersecting: 1 if either are 1, 0 if both are 0
seg_mat_combined = np.logical_or(seg_mat_paternal, seg_mat_maternal)

#seg_mat_combined = np.concatenate((seg_mat_paternal, seg_mat_maternal), axis=1)
npmi_combined = e2d.npmi_2d_fast(seg_mat_combined.astype(int), seg_mat_combined.astype(int))
npmi_combined = np.nan_to_num(npmi_combined)

# calculate npmi difference between combined and paternal, maternal
npmi_ratio = np.abs(npmi_paternal / npmi_maternal)
#npmi_ratio2 = np.abs(npmi_ratio - 1)
npmi_diff = np.abs(npmi_paternal - npmi_maternal)

# relative error
npmi_RE = np.abs(npmi_paternal - npmi_maternal) / np.nanmean([npmi_paternal, npmi_maternal], axis=0)

# maximum value across npmi matrices
vmax_npmi = np.nanmax([npmi_paternal, npmi_maternal, npmi_combined])

#plot.variability(npmi_maternal, 'M', 'NPMI', vmax=vmax_npmi, vmin=0)
#plot.variability(npmi_paternal, 'P', 'NPMI', vmax=vmax_npmi, vmin=0)
#plot.variability(npmi_combined, 'C', 'NPMI', vmax=vmax_npmi, vmin=0)
plot.variability(npmi_RE, 'Relative error (P, M)', 'Relative error', ratio=True)
#plot.variability(npmi_ratio, 'P/M NPMI phased difference', 'NPMI ratio (centered on 1)', center=1, ratio=True)
 


# %% Calculate all pairwise co-separation probabilities
'''Calculate all pairwise co-separation probabilities'''
probs_mat, ranges_mat, midpoints_mat, n_samples_mat = vrb.calculate_probs_mat(seg_mat_combined, npmi_combined)

plot.variability(n_samples_mat, "Number of samples with <50% overlap", "# slices", ratio=True, vmin=0)
# The interval midpoints are estimates of the true probability of co-separation for each pair
plot.variability(midpoints_mat, "Bootstrapped Wilson score", "midpoint")
# The interval in which the true probability lies with 95% confidence is given by the interval [midpoint - range/2, midpoint + range/2]
plot.variability(ranges_mat, "Bootstrapped Wilson score range", "range")



# %% Reduce probability vectors to find variability in single dataset
'''Reduce probability vectors to find variability in single dataset'''
mode = 'probability'
models = {
    #'A': seg_mats_A,
    #'B': seg_mats_B,
    #'C': seg_mats_C,
    #'D': seg_mats_D,

    #'P': seg_mat_paternal,
    #'M': seg_mat_maternal,
    #'U': seg_mat_unphased,
    'C': seg_mat_combined}

operations = {#'Covariance': 'cov', # good in entropy, higher sensitivity
              ##'Gini coefficient': 'gini', # good in entropy, okay in probability, best balance #2
              #'Coefficient of var inverse': 'cv', # good in entropy, okay (best) in probability, sensitive to outliers?
              #'Standart deviation': 'std', # good in entropy, high background
              ##'Variance': 'var', # okay in entropy #1
              'Mean': 'mean', 
              #'Skewness inverse': 'skewness', #3P?
              ##'Kurtosis': 'kurtosis',
              ##'MAD': 'mad',
              #'Mode': 'mode',
              #'Trimmed mean': 'trim_mean',
              #'BWMD': 'bwmd',
              ##'Winsor': 'winsor',
              #'IQR': 'iqr',
              #'Range': 'range',
              }

pearson_RE = []
pearson_ratio = []
pearson_diff = []
op = []

for m, (name, operation) in enumerate(operations.items()):
    #all_matrices = np.zeros((len(operations), seg_mat_combined.shape[0], seg_mat_combined.shape[0]))
    #for i, (label, model) in enumerate(models.items()):
    #    all_matrices[i] = reduce_all(model, operation, mode)
    if operation == 'cv' or operation == 'skewness':
        vmax = np.nanmax(vrb.reduce_all(probs_mat, operation)) # np.nanpercentile(all_matrices, 99)
        
    for i, (label, model) in enumerate(models.items()):
        if operation == 'cv' or operation == 'skewness':
            var_matrix = vmax - vrb.reduce_all(probs_mat, operation)
        else:
            var_matrix = vrb.reduce_all(probs_mat, operation)
        #plot.variability(e2d.npmi_2d_fast(model.astype(int), model.astype(int)), label, 'NPMI', None, 1, 0)
        #plot.variability(var_matrix, f"Dataset {label} {mode} vectors", name, vmax=np.nanpercentile(var_matrix, 99))
        plot.variability(var_matrix, f"Dataset {label}: variability approximation", name, center=0, ratio=True)
        #print(f"{name} for {mode} vectors of dataset {label}")
        #print(scoring(var_matrix))
        #print(np.nansum(var_matrix))

        pearson_RE.append(statfx.correlate_matrices(var_matrix, npmi_RE))
        pearson_ratio.append(statfx.correlate_matrices(var_matrix, npmi_ratio))
        pearson_diff.append(statfx.correlate_matrices(var_matrix, npmi_diff))
        op.append(name)
        data = {'Operation': op,
                'Cor. P,M relative error': pearson_RE,
                'Cor. P/M ratio': pearson_ratio,
                'Cor. P-M difference': pearson_diff}
        
standardized_var_matrix = statfx.standardize_vector(var_matrix.flatten()).reshape(var_matrix.shape)
standardized_npmi_RE = statfx.standardize_vector(npmi_RE.flatten()).reshape(npmi_RE.shape)
standardized_diff = statfx.relative_error(standardized_var_matrix, standardized_npmi_RE)
plot.variability(standardized_diff, 'RE vs VRB', 'Relative error', center=0, ratio=True)

scores = pd.DataFrame(data).sort_values('Cor. P,M relative error', key=abs, ascending=False)
scores



# %% Scan genome for highest correlation between npmi and variability
"""Scan genome for highest correlation between npmi and variability"""
import warnings

genome = {'chr1': 195}#, 'chr2': 181, 'chr3': 159, 'chr4': 156, 'chr5': 151, 'chr6': 149, 'chr7': 144, 'chr8': 130, 'chr9': 124, 'chr10': 130, 'chr11': 121, 'chr12': 120, 'chr13': 120, 'chr14': 125, 'chr15': 104, 'chr16': 98, 'chr17': 95, 'chr18': 90, 'chr19': 61}
window_size = 4
step_size = 2
mode = 'probability'
operation = 'mean'


current_data = []
for chromosome, length in genome.items():
    cors = []
    vrbs = []
    diffs = []
    windows = []
    props = []
    current_chr = []

    for window in tqdm(range(0, length, step_size)):
        seg_mat_paternal_window = select_region(seg_table_paternal, chromosome, window, window, window_size)
        seg_mat_maternal_window = select_region(seg_table_maternal, chromosome, window, window, window_size)
        seg_mat_combined_windows = np.logical_or(seg_mat_paternal_window, seg_mat_maternal_window)

        npmi_paternal_window = e2d.npmi_2d_fast(seg_mat_paternal_window.astype(int), seg_mat_paternal_window.astype(int))
        npmi_maternal_window = e2d.npmi_2d_fast(seg_mat_maternal_window.astype(int), seg_mat_maternal_window.astype(int))

        npmi_combined_windows = e2d.npmi_2d_fast(seg_mat_combined_windows.astype(int), seg_mat_combined_windows.astype(int))
        npmi_combined_windows = np.nan_to_num(npmi_combined_windows)

        # suppress runtime awrning from nanmean
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            npmi_RE_window = np.divide(np.abs(npmi_paternal_window - npmi_maternal_window), np.nanmean([npmi_paternal_window, npmi_maternal_window], axis=0), out=np.zeros_like(npmi_paternal_window), where=np.nanmean([npmi_paternal_window, npmi_maternal_window], axis=0)!=0)
        
            diff = np.nanmean(npmi_RE_window)
        #if diff < 0.25: continue

        # count NaNs in either npmi_paternal or npmi_maternal
        #diff_nans = np.abs(np.count_nonzero(np.isnan(npmi_paternal_window)) - np.count_nonzero(np.isnan(npmi_maternal_window))) / np.prod(npmi_RE_window.shape)
        proportion_nans = np.count_nonzero(np.isnan(npmi_RE_window)) / np.prod(npmi_RE_window.shape)
        #if proportion_nans > 0.02: continue
        props.append(proportion_nans)
        diffs.append(diff)

        vrb_matrix = vrb.reduce_all(seg_mat_combined_windows, operation, mode, npmi_combined_windows)
        vrb = np.nanmean(vrb_matrix)
        vrbs.append(vrb)
        cor = statfx.correlate_matrices(vrb_matrix, npmi_RE_window)
        cors.append(cor)

        current_chr.append(chromosome)
        windows.append(window)
        data = {'Chr': current_chr, 'Window': windows, 'Mean RE': diffs, 'Mean vrb': vrbs, 'Correlation': cors, 'Proportion non-zero': props}
    current_data.append(data)

# make current_data into one dataframe
df_genome_2 = pd.concat([pd.DataFrame(data) for data in current_data], ignore_index=True)
df_genome_2

# filter out rows that contain Nans
df_genome_fn = df_genome_2.dropna()

# filter rows with proportion of non-zero values > 0.02
df_genome_fn = df_genome_fn[df_genome_fn['Proportion non-zero'] < 0.02]

# perason correlation between variability and npmi relative error
cor_vrb_re = spearmanr(df_genome_fn['Mean vrb'], df_genome_fn['Mean RE'])[0]

# save df_genome_2
#df_genome_2.to_csv('df_genome_2.csv', index=False)

# bin Correlation values
df_genome_fn['Correlation bin'] = pd.cut(df_genome_fn['Correlation'], bins=10, labels=False)

import seaborn as sns
from matplotlib import pyplot as plt
sns.set_theme(style='whitegrid')
sns.scatterplot(data=df_genome_fn, x='Mean vrb', y='Mean RE', hue='Correlation bin', palette='viridis', alpha=1, s=10)
plt.show()



# %% Visualize distribution in selected probability vectors
"""Visualize distribution in selected probability vectors"""
# set matrix values to 0 based on number of samples or confidence interval ranges
filtered_midpoints_mat = midpoints_mat.copy()
#med_samples = np.nanpercentile(samples_mat, 25)
#filtered_midpoints_mat[samples_mat > med_samples] = 0
#med_range = np.percentile(ranges_mat[ranges_mat!=0], 75)
#filtered_midpoints_mat[ranges_mat > med_range] = 0

# specify pairs manually
#interesting_pairs = [(11, 58), (34, 73), (18,58), (43, 64), (46, 62), (4, 26), (70, 72), (40, 41), (2, 3)] # chr3 88
#interesting_pairs = [(31,123), (3,114), (13,37), (33,101), (108,130), (9,59), (46,95), (70,88), (40,48)]
#interesting_pairs =[(0,1), (2,3)]#, (40,41)]

# get pairs from variability analysis, convert to tuples and sort each tuple
interesting_pairs = select_interesting_pairs(midpoints_mat, 8)

#plot.variability(midpoints_mat, "Dataset C: variability approximation", operation, vmax=np.nanpercentile(midpoints_mat**5, 99.5), ratio=True, center=0)
plot.variability(filtered_midpoints_mat, f"Dataset C: variability approximation", 'Wilson', interesting_pairs)
#plot.variability(npmi_RE, 'Relative error (P, M)', 'Relative error', interesting_pairs, ratio=True, center=0)

# get vectors of selected pairs and plot their distributions
#probs_mat_dict = calculate_probs_mat(dataset, mode)
vectors = get_prob_vectors(interesting_pairs, probs_mat)

labels = [f"L {x}, {y}" for x, y in interesting_pairs]
colors = plot.get_pair_colors(midpoints_mat, interesting_pairs)
#centers = [vrb.reduce(vector, operation) for vector in vectors]
centers = [midpoints_mat[pair] for pair in interesting_pairs]

# scale npmi_ratio between 0 and 1
#standardized_npmi_RE = standardize_vector(npmi_RE.flatten()).reshape(npmi_RE.shape)
#npmi_RE_flat = np.clip(np.nan_to_num(npmi_RE).flatten(), a_min = 0, a_max=None) + 1
#log_trans_npmi_RE = np.log(np.log(npmi_RE_flat+1)).reshape(npmi_RE.shape)
#npmi_RE_norm = statfx.scale_matrix(log_trans_npmi_RE)
#npmi_diff = (npmi_diff - np.nanmin(npmi_diff)) / (np.nanmax(npmi_diff) - np.nanmin(npmi_diff))

# get npmi values for each pair
npmi_mat_pairs = []
npmi_pat_pairs =[] 
npmi_com_pairs = []
npmi_pm_ratios = []
npmi_pm_diffs = []
mids=[]
#for pair in interesting_pairs:
    #npmi_mat_pairs.append(npmi_maternal[pair[0], pair[1]])
    #npmi_pat_pairs.append(npmi_paternal[pair[0], pair[1]])
    #npmi_com_pairs.append(npmi_combined[pair[0], pair[1]])
    #npmi_pm_ratios.append(npmi_RE_norm[pair[0], pair[1]])
    #npmi_pm_diffs.append(npmi_diff[pair[0], pair[1]])
    #mids.append(midpoints[pair])

plot.ridge(vectors, centers, 'Weighted probability of locus to loci-pair co-separation', labels, colors, None, None, None, None)



# %% Compare measures using lineplot
import seaborn as sns
import matplotlib.pyplot as plt

def extract_upper_triangular(matrix):
    """Extract upper triangular of a matrix."""
    return matrix[np.triu_indices(matrix.shape[0], k=1)]

results = []
op = []
operations = {#'var': 'var', # okay in entropy #1
              #'sum': 'sum', 
              #'mean': 'mean', 
              #'skewness': 'skewness', #3P?
              #'kurtosis': 'kurtosis',
              #'mad': 'mad',
              'mode': 'mode',
              #'trim_mean': 'trim_mean',
              #'bwmd': 'bwmd',
              #'winsor': 'winsor',
              #'iqr': 'iqr',
              'range': 'range',
              #'std': 'std',
              }

for m, (name, operation) in enumerate(operations.items()):
    res = vrb.reduce_all(seg_mat_combined, operation, mode)
    results.append(extract_upper_triangular(res))
    op.append(name)

#results.append(extract_upper_triangular(npmi_ratio))
#op.append('NPMI ratio')
results.append(extract_upper_triangular(npmi_diff))
op.append('NPMI difference')

# create dataframe 
df_lineplot = pd.DataFrame(results).T
df_lineplot.columns = op

df_lineplot = df_lineplot.sort_values(by='NPMI difference').reset_index(drop=True)

# Calculate z-scores for each column
z_score_df = (df_lineplot - df_lineplot.mean()) / df_lineplot.std()

# Melt the normalized DataFrame to a long format for seaborn
z_score_df['index'] = df_lineplot.index
df_lineplot_normalized = pd.melt(z_score_df, id_vars=['index'], var_name='Operation', value_name='Values')

# Plot the lineplot using seaborn
plt.figure(figsize=(15, 5))
ax = sns.lineplot(data=df_lineplot_normalized, x='index', y='Values', hue='Operation', palette='tab10', legend='full', alpha=0.5, ci='sd')
ax.set_ylim([-3, 3])

# Move the legend outside the plot
ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.1))

plt.show()



# The difference or ratio in NPMI between two datasets describes the difference in association frequency between loci
# A high difference or ratio indicates that the loci pair are associated more often in one dataset than in the other

# Loci adjacency probability vectors describe the proportions of other loci to be found in the vicinity of a given locus pair
# A low probability vector indicates that other loci are not found together often with the given locus pair, suggesting that the given locus pair is part of a heterogeneous structure

