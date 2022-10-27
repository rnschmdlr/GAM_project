# %%
import sys
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import entropy_measures as em
import plotting.plot_large_data as plot


def load_data(filename):
    """Load segregation table from file and convert to numpy array"""
    col_to_skip = ['chrom', 'start', 'stop']
    segtab = pd.read_table(filename, usecols=lambda x: x not in col_to_skip)
    seg_mat = np.array(segtab, dtype=bool)

    return seg_mat


def region_entropy_plot(y, x, size, methods):
    #print('x:', y, 'to', y+size, '\ny:', x, 'to', x+size)
    if x < 0 or y < 0:
        print('no data below 3 Mb')
        sys.exit()

    offset = 60
    resolution = 0.05
    vmin = 0
    vmax = 1

    start = offset + min(x, y)
    end = start + size
    seg_mat_region = seg_mat[start:end, :]

    xticklabels = list(str(" ") * size)
    yticklabels = list(str(" ") * size)
    nth = int(size / 5)
    for idx, _ in enumerate(xticklabels):
        if not idx % nth:
            xticklabels[idx] = np.round(((y + offset + idx) * resolution), 2)
            yticklabels[idx] = np.round(((x + offset + idx) * resolution), 2)

    if x == y:
        x = 0
        y = 0
    if min(x, y) == x:
        x = 0
    else: 
        y = 0

    if 'npmi' in methods:
        #The pointwise mutual information represents a quantified measure for how much more- or less likely we are to see the two events co-occur, 
        # given their individual probabilities, and relative to the case where the two are completely independent.
        npmi_mat = em.npmi_2d_fast(seg_mat_region.astype(int), seg_mat_region.astype(int))
        plot.slow_raster(npmi_mat[x:x+size, y:y+size], vmin, vmax, xticklabels, yticklabels, title='Normalised Pointwise Mutual Information')

    if 'mi' in methods:
        je_mat = em.all_joint_entropy(seg_mat_region)
        mi_mat = em.mutual_information(je_mat)
        plot.slow_raster(mi_mat[x:x+size, y:y+size], vmin, vmax, xticklabels, yticklabels, title='Mutual Information')

        if 'rajski' in methods:
            # distance in information of the two partitions
            rajski_mat = 1 - mi_mat / je_mat
            plot.slow_raster(rajski_mat[x:x+size, y:y+size], vmin, vmax, xticklabels, yticklabels, title='Rajski distance (Nnrm. variation of information)')

    if 'ncmi' in methods:
        ncmi_mat = em.normalized_mutual_information(seg_mat_region)
        plot.slow_raster(ncmi_mat[x:x+size, y:y+size], vmin, vmax, xticklabels, yticklabels, title='Normalised Corrected Mutual Information')

    if 'vs' in methods:
        npmi_mat = em.npmi_2d_fast(seg_mat_region.astype(int), seg_mat_region.astype(int))
        ncmi_mat = em.normalized_mutual_information(seg_mat_region)

        # shows negative values in NPMI
        #ratio = (ncmi_mat - npmi_mat) / ncmi_mat
        #vmax = np.nanmax(ratio)
        #vmin = np.nancmin(ratio)

        # compares positive NPMI and ncmi
        #npmi_mat[npmi_mat < 0] = np.nan #postive NPMI
        #ratio = (npmi_mat - ncmi_mat) / npmi_mat

        ratio = np.abs(npmi_mat - ncmi_mat) / np.abs(npmi_mat)
        
        plot.slow_raster(ratio[x:x+size, y:y+size], vmin, vmax, xticklabels, yticklabels, title='npmi vs ncmi')

    if 'je' in methods:
        # amount by which the uncertainty of one random variable is reduced due to the knowledge of another
        je_mat = em.all_joint_entropy(seg_mat_region)
        mean_je_mat = np.mean(je_mat[x:x+size, y:y+size])
        #je_mat = scale(je_mat)
        #np.fill_diagonal(je_mat, 0)
        plot.slow_raster(je_mat[x:x+size, y:y+size], vmin, vmax, xticklabels, yticklabels, title='Joint Entropy')
        print('Average joint entropy:', np.round(mean_je_mat, 4))



def sliding_window_measures(segreg_data, start, end, step, size_range, methods):
    # data specific params
    offset = 3
    f = 1000000 / 50000 # million / resolution

    # convert Mb to xy coordinates
    a = int(start * f)
    b = int(end * f)
    x = int(step * f)
    
    visited = []
    wsize = []
    shaen = []
    cshaen = []
    sel = []
    je = []
    ije = []
    iqr = []
    rajski = []
    mi = []
    ncmi = []
    npmi = []

    for size in size_range:
         # check if size is even
        assert (size * f) % 2 == 0
        # check if windows touch
        assert step <= (size * 2)
        # check that first window is within bounds
        assert (start - size / 2) > offset

        s = int(size * f)
        r = s / 2

        for current in tqdm(range(a, b, x)):
            x0 = int(current - r)
            x1 = int(current + r)
            region = segreg_data[x0:x1, :]
            visited.append(current / f)
            wsize.append(s / f)

            if 'shaen' in methods:
                val = em.shannon_entropy(region)
                shaen.append(val)

            if 'cshaen' in methods:
                val = em.corrected_shannon_entropy(region)
                cshaen.append(val)
            
            if 'shaen_loci' in methods:
                val = em.shannon_entropy_multivar(region, 1)
                sel.append(np.mean(val))

            if 'je' in methods:
                je_mat = em.all_joint_entropy(region)
                je.append(np.mean(je_mat))

            if 'ije' in methods:
                je_mat = em.all_joint_entropy(region)
                mi_mat = em.mutual_information(je_mat)
                shaen_loci = np.mean(em.shannon_entropy_multivar(region, 1))
                val = np.mean(mi_mat) / (2 * shaen_loci)
                val = np.nan_to_num(val)
                ije.append(val)

            if 'mi' in methods:
                je_mat = em.all_joint_entropy(region)
                mi_mat = em.mutual_information(je_mat)
                mi.append(np.mean(mi_mat))

                if 'rajski' in methods:
                    # normalized variation of information
                    val = (np.mean(je_mat) - np.mean(mi_mat)) / np.mean(je_mat)
                    rajski.append(val)

                if 'iqr' in methods:
                    # information quality ratio
                    val = np.mean(mi_mat) / np.mean(je_mat)
                    val = np.nan_to_num(val)
                    iqr.append(val)

            if 'ncmi' in methods:
                ncmi_mat = em.normalized_mutual_information(region)
                ncmi.append(np.mean(ncmi_mat))

            if 'npmi' in methods:
                npmi_mat = em.npmi_2d_fast(region.astype(int), region.astype(int))
                npmi.append(np.mean(np.nan_to_num(npmi_mat)))

        d = {'visited': visited,
            'window size': wsize,
            'shaen': shaen, 
            'cshaen': cshaen,
            'shaen_loci': sel,
            'je': je, 
            'ije': ije,
            'iqr': iqr,
            'mi': mi, 
            'rajski': rajski,
            'ncmi': ncmi, 
            'npmi': npmi}
    
        d = dict([(k, v) for k, v in d.items() if len(v) > 0]) # remove entries with empty lists

    df = pd.DataFrame(d)

    return df


# %%
'''Load data'''
seg_mat = load_data('/Users/pita/Documents/Rene/GAM_project/data/Curated_GAM_mESCs_46C_real_1NP_at50000.passed_qc_fc5_cw6_s11.table')


# %% 
'''2D Cosegregation measure figure'''
xmb, ymb = 1302, 1302
size = 6
methods = ['ncmi', 'mi', 'rajski']
x = int(xmb * 100 / 5 - 60)
y = int(ymb * 100 / 5 - 60)
s = int(size * 100 / 5)
matrix = region_entropy_plot(x, y, s, methods)


# %%
'''Sliding window lineplot'''
start, stop, step = 1307, 1314, 1/5
size_range = [1,3,10]
methods = ['ncmi', 'npmi']
windows = sliding_window_measures(seg_mat, start, stop, step, size_range, methods)


#windows['ncmi'] = 1 - (1 - np.sqrt(windows['nmi']))**1.2315
#windows['vs'] = windows['npmi'] - windows['ncmi']
#windows['rasjki'] = 1 - windows['iqr']
windows2 = windows.drop(windows[windows['window size']<1].index)
windows2 = windows2.rename(columns={'ncmi': 'NCMI'})
windows2 = windows2.rename(columns={'npmi': 'NPMI'})
#windows2 = windows2.rename(columns={'cshaen': 'Shannon H'})
plot.lineplot(windows2, ['NCMI', 'NPMI'])


# %%
windows["je"] = (windows["je"] - min(windows["je"])) / max(windows["je"])
np.fill_diagonal(matrix, windows["je"].values)
plot.slow_raster(matrix[0:640, 0:640], 0, 1, "None", "None", title='Normalised Corrected Mutual Information')