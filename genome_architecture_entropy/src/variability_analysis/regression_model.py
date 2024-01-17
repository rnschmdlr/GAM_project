# %% Imports and helper function definitons
"""Imports and helper function definitons"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from variability_analysis.estimate_variability import reduce_all
from variability_analysis.statistic_functions import correlate_matrices
import variability_analysis.plotting as plot



def pool_matrix(matrix, pooling=8):
    # check if matrix can be pooled
    if matrix.shape[0] % pooling != 0:
        print('Matrix cannot be pooled')
        return matrix
    
    pool_matrix = np.zeros((int(matrix.shape[0] / pooling), int(matrix.shape[0] / pooling)))
    for i in range(int(matrix.shape[0] / pooling)):
        for j in range(int(matrix.shape[0] / pooling)):
            pool_matrix[i,j] = np.mean(matrix[i * pooling:(i+1) * pooling, j * pooling:(j+1) * pooling])
    return pool_matrix



# %% Load data
#TODO: load data from new data folder
seg_mat_combined = np.load('data/seg_mat_combined.npy')
npmi_RE = np.load('data/npmi_RE.npy')



# %% Prepare regression 
'''Prepare regression'''
mode = 'probability'
models = {
    #'A': seg_mats_A,
    #'B': seg_mats_B,
    #'C': seg_mats_C,
    #'D': seg_mats_D,

    #'M': seg_mat_maternal,
    #'P': seg_mat_paternal,
    #'U': seg_mat_unphased,
    'C': seg_mat_combined,
}

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

