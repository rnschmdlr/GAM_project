import numpy as np
from itertools import combinations
from scipy.spatial import distance
from scipy.stats import skew, kurtosis, trim_mean, mstats
from scipy.stats import norm as stats_norm
from scipy.stats import mode as modus
from multiprocessing import Pool
import multiprocessing ;multiprocessing.set_start_method('fork', force=True)

from entropy.shannon_entropies.compute_2d_entropies import shannon_entropy, shannon_entropy_axis
from variability_analysis.statistic_functions import gini_coefficient



def print_dimensions(nested_list):
    try:
        outer_length = len(nested_list)
        inner_length = len(nested_list[0])
        print(f"The outer list has {outer_length} elements.")
        print(f"The inner lists have {inner_length} elements.")
    except TypeError:
        print("The provided object is not a nested list.")
    except IndexError:
        print("The provided list is empty.")


def calculate_probability(matrix, pair, npmi_mat=None):
    """
    Calculate the probability of co-separation for a pair of loci.
    :param matrix: A numpy array with binary values where rows are loci and columns are samples.
    :param pair: A tuple of two loci to check.
    :return: A numpy array with the probability of occurrence for each locus.
    """
    def select_samples(similarity_matrix, max_dissimilarity_threshold=0.5):
        """
        Select a subset of samples from a similarity matrix using a greedy algorithm.
        :param similarity_matrix: A numpy array with pairwise similarities between samples.
        :param max_dissimilarity_threshold: The maximum dissimilarity allowed between the selected samples.
        :return: A set of indices of the selected samples.
        """
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

    #print('pair:', pair ,'mean:', np.mean(sim_matrix))

    # Select samples with with greedy algorithm to reduce similarity
    dissim_treshold = 0.5 #np.nanpercentile(distance.pdist(seg_mat_combined.T, 'hamming'), 90)
    samples_min_sim = select_samples(sim_matrix, dissim_treshold)
    # Subset matrix to samples with minimum similarity
    sub_sub_matrix = sub_matrix[:, list(samples_min_sim)]
    n_samples_viable = len(list(samples_min_sim))
    
    # Step 2: Check for each locus in these samples how often they are present.
    # The resulting sum is always a list even if there is only one locus.
    locus_counts = np.atleast_1d(np.sum(sub_sub_matrix, axis=1))
    # Remove self count
    locus_counts[pair[0]] = 0
    locus_counts[pair[1]] = 0

    if npmi_mat is not None:
        # Step 3: weigh loci counts by loci pair npmi values
        vmax = np.nanmax(npmi_mat)
        pair_contact_dist = np.nan_to_num(npmi_mat[pair[0], pair[1]], 0)
        locus_counts = locus_counts * (pair_contact_dist / vmax)

    # Step 4: Calculate the probability by dividing the number of times a locus is present 
    # in the set of samples by the number of samples
    probabilities = locus_counts / n_samples_viable
    
    return probabilities, n_samples_viable


def calculate_wilson_interval(samples):
    midpoints = []
    interval_ranges = []
    
    for sample in range(len(samples)):
        data = samples[sample]
        observed_probability = np.mean(data)
        
        if observed_probability >= 1:
            return (1, 1)
        
        # Calculate Wilson Score Interval
        confidence_level = 0.95
        n = len(data)
        z_score = stats_norm.ppf((1 + confidence_level) / 2)
        
        numerator = observed_probability + (z_score ** 2) / (2 * n)
        denominator = 1 + (z_score ** 2) / n
        margin_of_error = z_score * np.sqrt((observed_probability * (1 - observed_probability) / n) + (z_score ** 2) / (4 * (n ** 2)))
        
        lower_bound = (numerator - margin_of_error) / denominator
        upper_bound = (numerator + margin_of_error) / denominator
        
        midpoint = 1 - np.mean([lower_bound, upper_bound])
        interval_range = upper_bound - lower_bound

        midpoints.append(midpoint)
        interval_ranges.append(interval_range)
        
    return (midpoints, interval_ranges)


def wilson_score_interval_bootstrapped(probabilities):
    """
    This function performs a bootstrap analysis of the Wilson score interval for a given set of probabilities.

    It first generates a number of bootstrap samples from the input probabilities. Then, for each bootstrap sample, 
    it calculates the Wilson score interval. Finally, it calculates and returns the average midpoint and average 
    range of these intervals.

    This gives an estimate of the central tendency and variability of the Wilson score interval in the statistical 
    population from which the data sample was drawn.

    Parameters:
    probabilities (numpy.ndarray): An array of probabilities.

    Returns:
    tuple: A tuple containing the average midpoint and average range of the Wilson score intervals for the bootstrap samples.
    """

    # Manual bootstrapping
    n_bootstrap = 100
    size = 20
    if len(np.unique(probabilities)) == 1:
        if np.unique(probabilities)[0] == 0:
            # If all probabilities are zero, there are no valid intervals to calculate
            return (np.nan, np.nan)
        # If there's only one non-zero unique element in probabilities, replicate it
        bootstrap_samples = np.full((n_bootstrap, size), probabilities[0])
    else:
        # Otherwise, perform bootstrap sampling
        bootstrap_samples = np.random.choice(probabilities, size=(n_bootstrap, size), replace=True)
    
    # Calculate Wilson score interval for each bootstrap sample
    result = calculate_wilson_interval(bootstrap_samples)

    # join results
    midpoints = []
    ranges = []
    #for result in results:
    if isinstance(result[0], list):
        midpoints.extend(result[0])
    else:
        midpoints.append(result[0])
    if isinstance(result[1], list):
        ranges.extend(result[1])
    else:
        ranges.append(result[1])
    
    # Convert to numpy arrays
    midpoints = np.array(midpoints)
    ranges = np.array(ranges)

    # Calculate averages
    avg_midpoint = np.mean(midpoints)
    avg_interval_range = np.mean(ranges)

    return avg_midpoint, avg_interval_range


def calculate_entropy(matrix, pair):
    """
    Calculate the entropy of co-separation for a pair of loci.
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
    
    entropy = shannon_entropy_axis(sub_matrix, axis=1)
    return entropy


def calculate_probs(i, j, matrix, npmi_mat):
    probs, n_samples, *_ = calculate_probability(matrix, (i, j), npmi_mat)
    avg_midpoint, avg_interval_range = wilson_score_interval_bootstrapped(np.atleast_1d(probs))
    return i, j, probs, avg_interval_range, avg_midpoint, n_samples


def calculate_probs_mat(matrix, npmi_mat=None):
    num_loci = matrix.shape[0]
    probs_mat = [[[] for _ in range(num_loci)] for _ in range(num_loci)]
    ranges_mat = np.zeros((num_loci, num_loci))
    midpoints_mat = np.zeros((num_loci, num_loci))
    n_samples_mat = np.zeros((num_loci, num_loci))

    # Create a pool of workers
    with Pool() as pool:
        # Generate all pairs of loci
        pairs = list(combinations(range(num_loci), 2))
        # Calculate probabilities in parallel
        results = pool.starmap(calculate_probs, [(i, j, matrix, npmi_mat) for i, j in pairs])

    # Assign results
    for i, j, probs, avg_interval_range, avg_midpoint, n_samples in results:
        probs_mat[i][j] = probs
        ranges_mat[i, j] = avg_interval_range
        midpoints_mat[i, j] = avg_midpoint
        n_samples_mat[i, j] = n_samples

    return probs_mat, ranges_mat, midpoints_mat, n_samples_mat


def reduce(vector, operation, trim=0.25):
    """
    Reduce a vector to a single value using a given operation.
    :param vector: A numpy array with values.
    :param operation: A string indicating the operation to use.
    :param trim: A float indicating the percentage of values to trim from both ends of the vector if applicable.
    :return: A single integer."""

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
        return shannon_entropy(vector)
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