# %% Imports and helper function definitons
"""Imports and helper function definitons"""
import os
os.chdir('/Users/pita/Documents/Rene/GAM_project/genome_architecture_entropy/src/')

import numpy as np
from itertools import combinations
import entropy.shannon_entropies.compute_2d_entropies as e2d
import numpy as np
from scipy.spatial import distance
from scipy.stats import spearmanr, pearsonr, wasserstein_distance, skew, kurtosis, trim_mean, mstats
from scipy.stats import norm as stats_norm
from scipy.stats import mode as modus
from scipy.linalg import norm
import pandas as pd
from tqdm.auto import tqdm

import entropy.transfer_entropy.variability_plotting as plot


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

def euclidean_dist(p, q):
    # It's a straight-line distance between two points in Euclidean space.
    # Pros: Intuitive, widely used, and considers all dimensions.
    # Cons: Might be sensitive to magnitude changes in a specific dimension.
    return np.linalg.norm(p - q)

def manhattan_dist(p, q):
    # It's the distance between two points in a grid-based path.
    # Pros: Useful when considering grid-like problems.
    # Cons: Might be less intuitive than Euclidean for some scenarios.
    return distance.cityblock(p, q)

def cosine_similarity(p, q):
    # Measures the cosine of the angle between two non-zero vectors.
    # Pros: Can be useful when the magnitude of the vectors doesn't matter, and you're more interested in the direction (i.e., when you want to check if two vectors are pointing in the same direction).
    # Cons: Not a true "distance" metric in mathematical terms as it measures similarity.
    return np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))

def jensen_shannon_divergence(p, q, epsilon=1e-10):
    # It's a method of measuring the similarity between two probability distributions.
    # It's symmetric and is derived from the Kullback-Leibler divergence.
    # Pros: Gives a value between 0 (identical) and 1 (maximally different).
    # Cons: More computationally expensive than other methods.
    p = p + epsilon
    q = q + epsilon
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

def hellinger_distance(p, q):
    # Measures the similarity between two probability distributions.
    # It's related to the Bhattacharyya distance.
    # Pros: Bounded between 0 and 1, where 0 means no overlap and 1 means complete overlap.
    # Cons: Might not always capture subtle differences between distributions.
    return norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)

def total_var_distance(p, q):
    # It's a measure of the difference between two probability distributions.
    # Pros: Bounded and intuitive. It's half the L1 norm of the difference.
    # Cons: Might be less sensitive to changes in the tails of distributions.
    return 0.5 * np.sum(np.abs(p - q))

def pearsons_corr_coeff(p, q):
    # Measures the linear relationship between two datasets.
    # Pros: Results range between -1 and 1, providing a clear understanding of the relationship's direction and magnitude.
    # Cons: Only captures linear relationships.
    return 1-pearsonr(p, q)[0]

def spearmans_rank_corr_coeff(p, q):
    # It's a non-parametric version of the Pearson correlation, considering rank instead of actual values.
    # Pros: Can capture monotonic relationships which Pearson can't.
    # Cons: Assumes that the data is ordinal.
    return 1-spearmanr(p, q)[0]

def earth_movers_distance(p, q):
    # Measures the distance between two probability distributions over a region.
    # Pros: Can be more intuitive than other metrics as it's akin to "how much dirt do I need to move to make these piles the same?"
    # Cons: Computationally expensive.
    return wasserstein_distance(p, q)

def covariance(p, q):
    # It measures the joint variability of two random variables.
    # Pros: Provides direction of the relationship.
    # Cons: Doesn't normalize, so it's hard to interpret without additional context.
    return np.cov(p, q)[0, 1]

def mutual_information(p, q):
    # Measures the mutual dependence of two random variables.
    # mutual information between two vectors
    return e2d.normalized_mutual_information(np.column_stack((p, q)))[0,-1]

def npmi_wrapper(matrix):
    matrix = e2d.npmi_2d_fast(matrix.T.astype(int), matrix.T.astype(int))
    return matrix

def biweight_midvariance(data, c=9.0):
    # Central location (median)
    M = np.nanmedian(data)
    
    # Median Absolute Deviation (MAD)
    MAD = np.nanmedian(np.abs(data - M))
    
    # Ui values
    ui = (data - M) / (c * MAD)
    
    # Only consider data points where |ui| < 1
    mask = np.abs(ui) < 1
    
    # Biweight Midvariance Calculation
    numerator = np.nansum(((1 - ui[mask] ** 2) ** 2) * (data[mask] - M) ** 2)
    denominator = np.nansum((1 - ui[mask] ** 2) ** 2)
    
    # If denominator is zero, return zero to avoid division by zero
    if denominator == 0:
        return 0
    
    n = np.sum(~np.isnan(data)) # len of data without NaNs
    return n * (numerator / (denominator ** 2))



def calculate_probability(matrix, pair, npmi_mat=None):
    """
    :param matrix: A numpy array with binary values where rows are loci and columns are samples.
    :param pair: A tuple of two loci to check.
    :return: A numpy array with the probability of occurrence for each locus.
    """
    # Step 1: Identify all the samples where both loci from the pair are present.
    samples_with_both_loci = np.where((matrix[pair[0]] == 1) & (matrix[pair[1]] == 1))[0]
    # If no samples found with both loci, return a probability array with zeros.
    if len(samples_with_both_loci) == 0:
        return np.zeros(matrix.shape[0])

    # Subset matrix to samples with both loci
    sub_matrix = matrix[:, samples_with_both_loci]
    
    # Step 1.5: Reduce similarity between samples
    # Compute similarity matrix between all pairwise samples 
    sim_matrix = 1 - distance.squareform(distance.pdist(sub_matrix.T, 'hamming'))
    # Find global 75th percentile of similarity
    dissim_treshold = 0.5 #np.nanpercentile(distance.pdist(seg_mat_combined.T, 'hamming'), 90)

    #print('pair:', pair ,'mean:', np.mean(sim_matrix))

    # Select samples with with greedy algorithm to reduce similarity
    samples_min_sim = select_samples(sim_matrix, dissim_treshold)
    # Subset matrix to samples with minimum similarity
    sub_sub_matrix = sub_matrix[:, list(samples_min_sim)]
    n_samples_viable = len(list(samples_min_sim))
    
    # Step 2: Check for each locus in these samples how often they are present.
    locus_counts = np.sum(sub_sub_matrix, axis=1)
    # Remove self count
    locus_counts[pair[0]] = 0
    locus_counts[pair[1]] = 0

    if npmi_mat is not None:
        # Step 3: weigh loci counts by loci pair npmi values
        vmax = np.nanmax(npmi_mat)
        pair_contact_dist = np.nan_to_num(npmi_mat[pair[0], pair[1]], 0)
        locus_counts = locus_counts * (pair_contact_dist / vmax)

    # Step 4: Calculate the probability by dividing the number of times a locus is present 
    # in the set of samples by the number of samples and scale to its max count.
    probabilities = locus_counts / n_samples_viable
    if len((probabilities, n_samples_viable))!=2:
        return 0,0
    
    return probabilities, n_samples_viable



def select_samples(similarity_matrix, max_dissimilarity_threshold=0.5):
    selected_samples = set()
    all_samples = set(range(len(similarity_matrix)))

    #first_sample = np.argmax(np.sum(similarity_matrix, axis=1))
    #selected_samples.add(first_sample)

    while True:
        remaining_samples = all_samples - selected_samples
        if not remaining_samples:
            break  # All available samples have been selected

        # Find the sample that minimizes dissimilarity with the selected samples
        current_sample = min(remaining_samples, key=lambda x: sum(similarity_matrix[x][y] for y in selected_samples))

        # Calculate the dissimilarity with the current set of selected samples
        current_dissimilarity = sum(similarity_matrix[current_sample][y] for y in selected_samples) / (len(selected_samples) + 1)

        # Check if adding the current sample keeps dissimilarity below the threshold
        if current_dissimilarity < max_dissimilarity_threshold:
            selected_samples.add(current_sample)
        else:
            break

    return selected_samples



def wilson_score_interval_bootstrapped(probabilities, total_samples, confidence_level=0.95):
    n = total_samples
    z_score = stats_norm.ppf((1 + confidence_level) / 2)
    
    # Function to calculate Wilson Score Interval
    def calculate_wilson_interval(data):
        observed_probability = np.mean(data)
        numerator = observed_probability + (z_score ** 2) / (2 * n)
        denominator = 1 + (z_score ** 2) / n
        margin_of_error = z_score * np.sqrt((observed_probability * (1 - observed_probability) / n) + (z_score ** 2) / (4 * (n ** 2)))
        
        lower_bound = (numerator - margin_of_error) / denominator
        upper_bound = (numerator + margin_of_error) / denominator
        
        midpoint = np.mean([lower_bound, upper_bound])
        interval_range = upper_bound - lower_bound
        
        return midpoint, interval_range

    # Manual bootstrapping
    n_bootstrap = 1000
    if type(probabilities) != dict:
        size = 1
    else:
        size = len(probabilities)
    bootstrap_samples = [np.random.choice(probabilities, size=size, replace=True) for _ in range(n_bootstrap)]

    # Calculate Wilson Score Interval for each bootstrap sample
    intervals = [calculate_wilson_interval(sample) for sample in bootstrap_samples]

    # Calculate average midpoint and average interval range
    midpoints = [interval[0] for interval in intervals]
    interval_ranges = [interval[1] for interval in intervals]

    avg_midpoint = np.mean(midpoints)
    avg_interval_range = np.mean(interval_ranges)

    return avg_midpoint, avg_interval_range




def calculate_entropy(matrix, pair):
    """
    :param matrix: A numpy array with binary values where rows are loci and columns are samples.
    :param pair: A tuple of two loci to check.
    :return: A numpy array with the entropies of occurrence for each locus.
    """
    # Step 1: Identify all the samples where both loci from the pair are present.
    samples_with_both_loci = np.where((matrix[pair[0]] == 1) & (matrix[pair[1]] == 1))[0]

    # If no samples found with both loci, return a probability array with zeros.
    if len(samples_with_both_loci) == 0:
        return np.zeros(matrix.shape[0])

    # Step 2: Check for each locus in these samples how often they are present.
    # Remove the loci pair from the matrix.
    mask = np.ones(matrix.shape[0], dtype=bool)
    mask[pair[0]] = False
    mask[pair[1]] = False
    sub_matrix = matrix[mask][:, samples_with_both_loci]
    
    entropy = e2d.shannon_entropy_axis(sub_matrix, axis=1)
    return entropy



def calculate_all_pairs(matrix, mode='probability', npmi_mat=None):
    num_loci = matrix.shape[0]
    all_pairs = {}
    ranges = {}
    midpoints = {}
    samples = {}
    val = 0

    # Iterate over all possible pairs of loci (upper triangle)
    for pair in combinations(range(num_loci), 2):
        if mode == 'probability':
            val, n_samples, *_ = calculate_probability(matrix, pair, npmi_mat)
            #avg_midpoint, avg_interval_range = wilson_score_interval_bootstrapped(val, n_samples)
        if mode == 'entropy':
            val = calculate_entropy(matrix, pair)
        all_pairs[pair] = val
        #ranges[pair] = avg_interval_range
        #midpoints[pair] = avg_midpoint
        samples[pair] = n_samples

    return all_pairs, ranges, midpoints, samples


def reduce(vector, operation, trim=0.25):
    if operation == 'mean':
        return np.nanmean(vector)
    if operation == 'var':
        return np.nansum(np.abs(vector - np.nanmean(vector)))
    if operation == 'sum':
        return np.nansum(vector)
    if operation == 'cov':
        return np.cov(vector)
    if operation == 'gini':
        return gini_coefficient(vector)
    if operation == 'skewness':
        return skew(vector)
    if operation == 'kurtosis':
        return kurtosis(vector)
    if operation == 'mad':
        return np.nanmedian(np.abs(vector - np.nanmedian(vector)))
    if operation == 'std':
        return np.nanstd(vector)
    if operation == 'cv':
        mean_val = np.nanmean(vector)
        if mean_val == 0:
            return 0
        return 100 * np.nanstd(vector) / mean_val
    if operation == 'entropy':
        return e2d.shannon_entropy(vector)
    if operation == 'mode':
        return modus(vector, keepdims=False)[0]
    if operation == 'trim_mean':
        return trim_mean(vector, trim)
    if operation == 'bwmd':
        return biweight_midvariance(vector)
    if operation == 'winsor':
        return np.nanmean(mstats.winsorize(vector, limits=[trim, trim]))
    if operation == 'iqr':
        return np.nanpercentile(vector, 75) - np.nanpercentile(vector, 25)
    if operation == 'range':
        return np.nanmax(vector) - np.nanmin(vector)
    

def reduce_all(all_pairs, operation):
    n_loci = all_pairs[max(all_pairs.keys())].shape[0]
    var_matrix = np.zeros((n_loci, n_loci))

    for pair in all_pairs:
        prob_vector = all_pairs[pair]
        var = reduce(prob_vector, operation)
        var_matrix[pair[0], pair[1]] = var
        var_matrix[pair[1], pair[0]] = var

    return var_matrix


def compute_difference(method, dataset1, dataset2, mode='probability'):
    all_pairs_1 = calculate_all_pairs(dataset1, mode)
    all_pairs_2 = calculate_all_pairs(dataset2, mode)

    num_loci = dataset1.shape[0]
    difference_matrix = np.zeros((num_loci, num_loci))

    for pair in all_pairs_1:
        prob_vector_1 = all_pairs_1[pair]
        prob_vector_2 = all_pairs_2[pair]
        
        diff = method(prob_vector_1, prob_vector_2)
        
        difference_matrix[pair[0], pair[1]] = diff
        difference_matrix[pair[1], pair[0]] = diff  # since it's symmetric
    return difference_matrix


def standardize_vector(vector):
    """Standardize a vector."""
    return (vector - np.nanmean(vector)) / (np.nanstd(vector) + 1e-10)


def scoring(matrix):
    summed_scores = np.nansum(matrix, axis=1)
    standardized_scores = standardize_vector(summed_scores)
    print('Gini coeff:', np.round(gini_coefficient(standardized_scores), 3))
    
    # Get the indices of the top N loci and extract their scores
    top_indices = np.argsort(standardized_scores)[::-1]
    top_scores = standardized_scores[top_indices]
    df_scores = pd.DataFrame({'locus': top_indices, 'score': top_scores})
    df_scores.index += 1

    return df_scores.head(5)


def gini_coefficient(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements

    return ((np.nansum((2 * index - n  - 1) * array)) / (n * np.nansum(array))) #Gini coefficient


def get_prob_vectors(loci_pairs_to_extract, prob_dict):
    """Get specific probability vectors based on loci pair numbers from a dictionary."""
    return [prob_dict[pair] for pair in loci_pairs_to_extract]


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


def pool_matrix(matrix, pooling=8):
    # check if matrix can be pooled
    if matrix.shape[0] % pooling != 0:
        print('Matrix cannot be pooled')
        return matrix
    
    pool_matrix = np.zeros((int(seg_mat_combined.shape[0]/pooling), int(seg_mat_combined.shape[0]/pooling)))
    for i in range(int(seg_mat_combined.shape[0]/pooling)):
        for j in range(int(seg_mat_combined.shape[0]/pooling)):
            pool_matrix[i,j] = np.mean(matrix[i*pooling:(i+1)*pooling, j*pooling:(j+1)*pooling])
    return pool_matrix


def correlate_matrices(mat1, mat2):
    """Correlate two matrices by using both Pearson and Cosine Similarity."""
    # make diagnonal NaNs
    np.fill_diagonal(mat1, np.nan)
    np.fill_diagonal(mat2, np.nan)

    # substract their mean
    #mat1 = mat1 - np.nanmean(mat1)
    #mat2 = mat2 - np.nanmean(mat2)

    # Flatten matrices
    flat_mat1 = mat1.flatten()
    flat_mat2 = mat2.flatten()

    # Remove NaNs
    mask = np.logical_and(~np.isnan(flat_mat1), ~np.isnan(flat_mat2))
    flat_mat1 = flat_mat1[mask]
    flat_mat2 = flat_mat2[mask]

    # Calculate correlation
    corr_p = spearmanr(flat_mat1, flat_mat2)[0]
    #plt.figure(figsize=(10, 10))
    #plt.scatter(flat_mat1, flat_mat2, s=5)
    #plt.show()

    return corr_p


def relative_error(mat1, mat2):
    """Calculate the relative error between two matrices."""
    # Calculate relative error
    rel_err = np.abs(mat1 - mat2) / np.nanmean([mat1, mat2], axis=0)
    return rel_err


def scale_matrix(mat):
    """Scale a matrix between 0 and 1."""
    return (mat - np.nanmin(mat)) / (np.nanmax(mat) - np.nanmin(mat))



# %% Example
matrix = np.array([[0, 1, 0, 1],
                   [1, 1, 0, 1],
                   [0, 1, 0, 0],
                   [1, 0, 1, 0],
                   [0, 0, 1, 1]])
pair = (1, 3)

#print(calculate_probability(matrix, pair))
all_pair_probabilities = calculate_all_pairs(matrix)

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
chrom = 'chr5'
x,y = 20,20
s = 8 #2.8
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
del data_paternal, data_maternal


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

plot.variability(npmi_maternal, 'M', 'NPMI', vmax=vmax_npmi, vmin=0)
plot.variability(npmi_paternal, 'P', 'NPMI', vmax=vmax_npmi, vmin=0)
plot.variability(npmi_combined, 'C', 'NPMI', vmax=vmax_npmi, vmin=0)
plot.variability(npmi_RE, 'Relative error (P, M)', 'Relative error', ratio=True, center=0)
#plot.variability(npmi_ratio, 'P/M NPMI phased difference', 'NPMI ratio (centered on 1)', center=1, ratio=True)
 


# %% Calculate all pairwise co-separation probabilities
'''CALCULATE ALL PAIRWISE CO-SEPARATION PROBABILITES'''
all_pairs, ranges, midpoints, samples = calculate_all_pairs(seg_mat_combined, 'probability', npmi_combined)

n_loci = all_pairs[max(all_pairs.keys())].shape[0]
samples_mat = np.zeros((n_loci, n_loci))
midpoints_mat = np.zeros((n_loci, n_loci))
ranges_mat = np.zeros((n_loci, n_loci))

for pair in all_pairs:
    rang = ranges[pair]
    mid = midpoints[pair]
    n_samples = samples[pair]
    samples_mat[pair[0], pair[1]] = n_samples
    midpoints_mat[pair[0], pair[1]] = mid
    ranges_mat[pair[0], pair[1]] = rang

plot.variability(samples_mat, "Dataset C: number of viable samples", "#", center=0, ratio=True)
plot.variability(midpoints_mat, "Dataset C: variability approximation", "bootstrap interval midpoints", center=0, ratio=True)
plot.variability(ranges_mat, "Dataset C: variability approximation", "bootstrap interval ranges", center=0, ratio=True)



# %% Comparing loci probability vector across datasets
"""Comparing loci probability vector across datasets"""
mode = 'probability'
models = {
    #'A': seg_mats_A,
    #'B': seg_mats_B,
    #'C': seg_mats_C,
    #'D': seg_mats_D,

    'M': seg_mat_maternal,
    'P': seg_mat_paternal,
    #'U': seg_mat_unphased,
    #'C': seg_mat_combined,
}

methods = {
    'Sum': lambda p, q: np.sum(np.abs(p - q)),
    'NPMI': lambda p, q: e2d.npmi_2d_fast(p.reshape(1, -1), q.reshape(1, -1)), # probability
    'IQR': lambda p, q: np.nanpercentile(p, 75) - np.nanpercentile(p, 25), # probability
    #'1-Spearman\'s Rank Correlation Coefficient': spearmans_rank_corr_coeff, # less noisy using probability
    #'1-Cosine Similarity': cosine_similarity,
    #'Jensen-Shannon Divergence': jensen_shannon_divergence,
    #'Euclidean Distance': euclidean_dist,
    #'Manhattan Distance': manhattan_dist,
    #'Hellinger Distance': hellinger_distance,
    #'Total var Distance': total_var_distance,
    #'Pearson\'s Correlation Coefficient': pearsons_corr_coeff,
    #'Earth Mover\'s Distance': earth_movers_distance,
    #'Covariance': covariance
    #'Cross entropy'
}

for m, (name, method) in enumerate(methods.items()):
    for i, (label1, model1) in enumerate(models.items()):
        for j, (label2, model2) in enumerate(models.items()):
            if j <= i:  # Skip combinations already computed or same model combinations
                continue
            if (j == 1 and i == 0) or (j == 3 and i == 2):
                difference_matrix = compute_difference(method, model1, model2, mode)
                print(f"{name} between {label1} and {label2}")
                print(scoring(difference_matrix))
                plot.vector_difference(name, difference_matrix, None, mode)

# NPMI interpretation of given pair of loci
# close to 0: distributions are relatively independent or less associated in each dataset. 
# => variability observed in one dataset does not provide much information about the variability in the other dataset



"""Reduce probability vectors to find variability in single dataset"""
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
        vmax = np.nanmax(reduce_all(all_pairs, operation)) # np.nanpercentile(all_matrices, 99)
        
    for i, (label, model) in enumerate(models.items()):
        if operation == 'cv' or operation == 'skewness':
            var_matrix = vmax - reduce_all(all_pairs, operation)
        else:
            var_matrix = reduce_all(all_pairs, operation)
        #plot.variability(e2d.npmi_2d_fast(model.astype(int), model.astype(int)), label, 'NPMI', None, 1, 0)
        #plot.variability(var_matrix, f"Dataset {label} {mode} vectors", name, vmax=np.nanpercentile(var_matrix, 99))
        plot.variability(var_matrix, f"Dataset {label}: variability approximation", name, center=0, ratio=True)
        #print(f"{name} for {mode} vectors of dataset {label}")
        #print(scoring(var_matrix))
        #print(np.nansum(var_matrix))

        #plot.variability(sample_mean, f"Dataset {label}: mean similarity", 'sim')
        #plot.variability(sample_var, f"Dataset {label}: variance similarity", 'sim')
        

        cor_RE = correlate_matrices(var_matrix, npmi_RE)
        cor_ratio = correlate_matrices(var_matrix, npmi_ratio)
        cor_diff = correlate_matrices(var_matrix, npmi_diff)
        pearson_RE.append(cor_RE)
        pearson_ratio.append(cor_ratio)
        pearson_diff.append(cor_diff)
        op.append(name)
        data = {'Operation': op,
                'Cor. P,M relative error': pearson_RE,
                'Cor. P/M ratio': pearson_ratio,
                'Cor. P-M difference': pearson_diff}
        
scores = pd.DataFrame(data).sort_values('Cor. P,M relative error', key=abs, ascending=False)
scores


standardized_var_matrix = standardize_vector(var_matrix.flatten()).reshape(var_matrix.shape)
standardized_npmi_RE = standardize_vector(npmi_RE.flatten()).reshape(npmi_RE.shape)
vmax_diff = max(np.nanpercentile(standardized_var_matrix, 99), np.nanpercentile(standardized_npmi_RE, 99))
standardized_diff = relative_error(standardized_var_matrix, standardized_npmi_RE)
plot.variability(standardized_diff, 'RE vs VRB', 'Relative error', center=0, ratio=True)

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

        vrb_matrix = reduce_all(seg_mat_combined_windows, operation, mode, npmi_combined_windows)
        vrb = np.nanmean(vrb_matrix)
        vrbs.append(vrb)
        cor = correlate_matrices(vrb_matrix, npmi_RE_window)
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
mode = 'probability'
operation = 'mean'
var_matrix = reduce_all(all_pairs, operation)

# set matrix values to 0 based on number of samples or confidence interval ranges
filtered_var_matrix = var_matrix.copy()
#med_samples = np.nanpercentile(samples_mat, 25)
#filtered_var_matrix[samples_mat > med_samples] = 0
#med_range = np.percentile(ranges_mat[ranges_mat!=0], 75)
#filtered_var_matrix[ranges_mat > med_range] = 0

# specify pairs manually
#interesting_pairs = [(11, 58), (34, 73), (18,58), (43, 64), (46, 62), (4, 26), (70, 72), (40, 41), (2, 3)] # chr3 88
#interesting_pairs = [(31,123), (3,114), (13,37), (33,101), (108,130), (9,59), (46,95), (70,88), (40,48)]
#interesting_pairs =[(0,1), (2,3)]#, (40,41)]

# get pairs from variability analysis, convert to tuples and sort each tuple
interesting_pairs = select_interesting_pairs(np.abs(standardized_diff), 8)

#plot.variability(var_matrix, "Dataset C: variability approximation", operation, vmax=np.nanpercentile(var_matrix**5, 99.5), ratio=True, center=0)
plot.variability(filtered_var_matrix, f"Dataset C: variability approximation", operation, interesting_pairs)
#plot.variability(npmi_RE, 'Relative error (P, M)', 'Relative error', interesting_pairs, ratio=True, center=0)

# get vectors of selected pairs and plot their distributions
#all_pairs_dict = calculate_all_pairs(dataset, mode)
vectors = get_prob_vectors(interesting_pairs, all_pairs)

labels = [f"L {x}, {y}" for x, y in interesting_pairs]
colors = plot.get_pair_colors(var_matrix, interesting_pairs)
centers = [reduce(vector, operation) for vector in vectors]

# scale npmi_ratio between 0 and 1
#standardized_npmi_RE = standardize_vector(npmi_RE.flatten()).reshape(npmi_RE.shape)
npmi_RE_flat = np.clip(np.nan_to_num(npmi_RE).flatten(), a_min = 0, a_max=None) + 1
log_trans_npmi_RE = np.log(np.log(npmi_RE_flat+1)).reshape(npmi_RE.shape)
npmi_RE_norm = scale_matrix(log_trans_npmi_RE)
#npmi_diff = (npmi_diff - np.nanmin(npmi_diff)) / (np.nanmax(npmi_diff) - np.nanmin(npmi_diff))

# get npmi values for each pair
npmi_mat_pairs = []
npmi_pat_pairs =[] 
npmi_com_pairs = []
npmi_pm_ratios = []
npmi_pm_diffs = []
mids=[]
for pair in interesting_pairs:
    #npmi_mat_pairs.append(npmi_maternal[pair[0], pair[1]])
    #npmi_pat_pairs.append(npmi_paternal[pair[0], pair[1]])
    #npmi_com_pairs.append(npmi_combined[pair[0], pair[1]])
    npmi_pm_ratios.append(npmi_RE_norm[pair[0], pair[1]])
    #npmi_pm_diffs.append(npmi_diff[pair[0], pair[1]])
    #mids.append(midpoints[pair])

plot.ridge(vectors, centers, 'Weighted probability of locus to loci-pair co-separation', labels, colors, None, None, None, npmi_pm_ratios)



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
    res = reduce_all(seg_mat_combined, operation, mode)
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


# %% Prepare regression 
'''Prepare regression'''
import seaborn as sns
import matplotlib.pyplot as plt

operations = {
              'Covariance': 'cov', # good in entropy, higher sensitivity
              ##'Gini coefficient': 'gini', # good in entropy, okay in probability, best balance #2
              #'Coefficient of var': 'cv', # good in entropy, okay (best) in probability, sensitive to outliers?
              #'std': 'std', # good in entropy, high background
              #'var': 'var', # okay in entropy #1
              #'mean': 'mean', 
              #'skewness': 'skewness', #3P?
              #'kurtosis': 'kurtosis',
              #'mad': 'mad',
              #'mode': 'mode',
              'trim_mean': 'trim_mean',
              #'bwmd': 'bwmd',
              #'winsor': 'winsor',
              #'iqr': 'iqr',
              'range': 'range',
              }

# Load your measurement and ground truth matrices
all_matrices = np.zeros((len(operations), seg_mat_combined.shape[0], seg_mat_combined.shape[0]))
for i, (label, model) in enumerate(models.items()):
    for m, (name, operation) in enumerate(operations.items()):
        if operation == 'skewness':
            vmax = np.nanmax(reduce_all(model, operation, mode))
            var_matrix = vmax - reduce_all(model, operation, mode)
        all_matrices[m] = reduce_all(model, operation, mode)

# Flatten the matrices
measurement, x, y = all_matrices.shape
X = all_matrices.reshape(measurement, -1).T
#set nans to 0
y = npmi_RE.flatten()
y = np.nan_to_num(y)

# Correlation analysis
correlation_matrix = np.corrcoef(X, rowvar=False)
labels = [f"{name}" for name, operation in operations.items()]
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, xticklabels=labels, yticklabels=labels)
plt.show()



# %% Regression Analysis
"""Regression Analysis"""
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

# Split data into training and testing sets
m = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scores = cross_val_score(m, X_train, y_train, scoring='neg_mean_squared_error', cv=4)

# repeat n times, only keep the model with the lowest MSE
best_mse = float('inf')
best_reg = None
best_y_pred = None
best_scores = None

for i in range(10000):
    reg = m.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    if mse < best_mse:
        best_mse = mse
        best_reg = reg
        best_y_pred = y_pred
        best_scores = scores


ops = list(operations.keys())
print('The target is: NPMI P/M Ratio')
for i,j in enumerate(best_reg.coef_):
    print(f"Feature {ops[i]}: Score: {j}")

print('Coefficient of determination: %.2f' % r2_score(y_test, best_y_pred))
print('Variance of target:', np.round(np.var(y), 3))

# Evaluate on the test set
print(f"MSE on the test set: {best_mse:.3f}")
print(f"Cross-validation MSE on training set: {-best_scores.mean():.3f} (+/- {best_scores.std():.3f})")



# %% Regression evaluation
result_matrix = np.zeros((len(operations), seg_mat_combined.shape[0], seg_mat_combined.shape[0]))
for j in range(len(operations)):
    result_matrix[j] = all_matrices[j] * best_reg.coef_[j]
result_matrix = np.sum(result_matrix, axis=0)
print(correlate_matrices(result_matrix, npmi_RE))

loci_scores_res = np.nansum(result_matrix, axis=1)
loci_scores_RE = np.nansum(npmi_RE, axis=1)
print(pearsonr(loci_scores_res, loci_scores_RE)[0])

# plot raw
plot.variability(result_matrix, 'Variability Regression Model', 'Probability', vmin=0)
plot.variability_ratio(result_matrix**2, 'Variability Regression Model', 'Probability', 0)
plot.variability_ratio(npmi_RE, 'P/M NPMI relative error (pooled)', 'Abs. distance to ratio equal 1', center=0)

# plot pooled
plot.variability_ratio(pool_matrix(npmi_RE, 4), 'P/M NPMI relative error (pooled)', 'Abs. distance to ratio equal 1', center=0)
plot.variability_ratio(pool_matrix(result_matrix**3, 4), 'Variability Regression Model (pooled)', 'Probability', 0)

