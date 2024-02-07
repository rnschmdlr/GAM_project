# %% Imports and helper function definitons
"""Imports, helper function definitons and data loading"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

# makes absolute imports possible for interactive use in different environments
# $ export GAE_PACKAGE_PATH=/path/to/genome_architecture_entropy
package_path = os.getenv('GAE_PACKAGE_PATH')
sys.path.insert(0, package_path) 

from src.entropy.shannon_entropies import compute_2d_entropies as e2d
from src.variability_analysis import estimate_variability as vrb


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


def get_chromosome_len(seg_table, res_kb=50):
    # print size of all individual chromosomes
    chromosomes = seg_table['chrom'].unique()
    chromosomes = np.array(sorted(chromosomes, key=chromosome_sort))
    chr_len = {}
    for chr in chromosomes:
        chr_len[chr] = seg_table[seg_table['chrom'] == chr].shape[0] * res_kb / 1000
    return chr_len


def chromosome_variability(seg_table, window_size, res_kb=50):
    """Calculate variability for each chromosome"""
    chr_len = get_chromosome_len(seg_table, res_kb)
    step_size = int(window_size / 2)
    data = []

    for n, (chr, chr_size) in enumerate(chr_len.items()):
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

    df_genome = pd.concat([pd.DataFrame(entry) for entry in data], ignore_index=True)
    df_genome['Chr'] = df_genome['Chr'].astype('category')
    return df_genome


if __name__ == '__main__':
    data_path = os.path.join(package_path, 'data/experimental/')
    data_file = os.path.join(data_path, 'Curated_GAM_mESCs_46C_real_1NP_at50000.passed_qc_fc5_cw6_s11.table')
    seg_table = pd.read_table(data_file)
    window_size = 4
    df_genome = chromosome_variability(seg_table, window_size)
    df_genome.to_csv('variability_per_chromosome.csv', index=False)
    print('Variability per chromosome saved to variability_per_chromosome.csv')
