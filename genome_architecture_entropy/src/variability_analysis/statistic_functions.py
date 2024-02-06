import numpy as np
import pandas as pd
from scipy.stats import spearmanr



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


def correlate_matrices(mat1, mat2):
    """Correlate two matrices by using both Spearman r."""
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


def differential_contacts(mat1, mat2):
    """Calculate the differential contacts between two matrices."""
    # Calculate zscores
    zscores1 = (mat1 - np.nanmean(mat1)) / np.nanstd(mat1)
    zscores2 = (mat2 - np.nanmean(mat2)) / np.nanstd(mat2)

    # Calculate differential contacts
    diff_contacts = np.abs(zscores1 - zscores2)
    return diff_contacts