import numpy as np
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr, wasserstein_distance, norm

from entropy.shannon_entropies import normalized_mutual_information


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
    return normalized_mutual_information(np.column_stack((p, q)))[0,-1]
