# %% Imports and helper function definitons
"""Imports, helper function definitons and data loading"""
import os
os.chdir('/Users/pita/Documents/Rene/GAM_project/genome_architecture_entropy/src/')

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import re

import variability_analysis.estimate_variability as vrb
import entropy.shannon_entropies.compute_2d_entropies as e2d


def select_region(data, chr, xmb, ymb, size, res_kb=50):
    seg_table_chr = data[data['chrom'] == chr]
    seg_table_chr = seg_table_chr.drop(columns=['chrom', 'start', 'stop'])
    seg_mat = np.array(seg_table_chr, dtype=bool)

    # base region to matrix coordinates
    x = int(xmb * 1000 / res_kb)
    y = int(ymb * 1000 / res_kb)
    s = int(size * 1000 / res_kb)

    # conversion to slices
    start = 0 + min(x, y)
    end = start + s
    seg_mat_region = seg_mat[start:end, :]
    
    return seg_mat_region


def chromosome_sort(chromosomes):
    match = re.match(r"(\D+)(\d+)?", chromosomes)
    if match:
        prefix, number = match.groups()
        return (prefix, int(number) if number else float('inf'))
    return chromosomes


path = '/Users/pita/Documents/Rene/GAM_project/genome_architecture_entropy/data/experimental/'
data = path + 'Curated_GAM_mESCs_46C_real_1NP_at50000.passed_qc_fc5_cw6_s11.table'
seg_table = pd.read_table(data)
res_kb = 50

# print size of all individual chromosomes
chromosomes = seg_table['chrom'].unique()
chromosomes = np.array(sorted(chromosomes, key=chromosome_sort))
for chr in chromosomes:
    print(chr, seg_table[seg_table['chrom'] == chr].shape[0]* res_kb / 1000)



# %% Calculate variability for each chromosome
"""Calculate variability for each chromosome"""
data = []
window_size = 4
step_size = int(window_size / 2)

for chr in chromosomes:
    chr_size = seg_table[seg_table['chrom'] == chr].shape[0] * res_kb / 1000
    # initialize lists
    mean_range = []
    mean_vrb = []
    var_vrb = []
    windows = []
    current_chr = []

    for window in tqdm(np.arange(0, chr_size, step_size), desc=f'{chr}'):
        # subset the data
        seg_mat_window = select_region(seg_table, chr, window, window, step_size, res_kb)
        npmi_window = e2d.npmi_2d_fast(seg_mat_window.astype(int), seg_mat_window.astype(int))
        
        # calculate variability estimated by the Wilson score
        _, ranges_mat, midpoints_mat, _ = vrb.calculate_probs_mat(seg_mat_window, npmi_window)

        # append results to lists
        current_chr.append(chr)
        mean_vrb.append(np.nanmean(midpoints_mat))
        var_vrb.append(np.nanvar(midpoints_mat))
        mean_range.append(np.nanmean(ranges_mat))
        current_data = {'Chr': current_chr, 'Variability est.': mean_vrb, 'Range': mean_range, 'Var': var_vrb}
        data.append(current_data)

# make current_data into one dataframe
df_genome = pd.concat([pd.DataFrame(entry) for entry in data], ignore_index=True)
# aggregate for each chromosome
df_genome.groupby('Chr').mean()

# save df_genome_2
df_genome.to_csv('df_genome_scan_iza_ws4_inverse.csv', index=False)

# drop rows where variability is zero
#df_genome = df_genome[df_genome['Variability est.'] > 0]

# %% Data visualization and statistics
# violin plots for each chromosome
import seaborn as sns
import matplotlib.pyplot as plt
#chromosomes = ['chr1', 'chr19', 'chrX']
#df_genome = pd.read_csv('df_genome_scan_iza_ws4.csv')
df_genome['Chr'] = df_genome['Chr'].astype('category')
df_genome['Chr'] = df_genome['Chr'].cat.set_categories(chromosomes)
df_genome = df_genome.sort_values('Chr')
plt.figure(figsize=(15, 6))
sns.violinplot(x='Chr', y='Variability est.', data=df_genome)
plt.xticks(rotation=90)
plt.ylim(0.25, 0.4)

# boxplots for each chromosome
plt.figure(figsize=(15, 6))
sns.boxplot(x='Chr', y='Variability est.', data=df_genome)
plt.xticks(rotation=90)
plt.ylim(0.25, 0.4)

# significance test between chrX and all other chromosomes
from scipy.stats import mannwhitneyu
chrX = df_genome[df_genome['Chr'] == 'chrX']['Variability est.']
for chr in chromosomes:
    if chr != 'chrX':
        print(chr, mannwhitneyu(chrX, df_genome[df_genome['Chr'] == chr]['Variability est.']))


