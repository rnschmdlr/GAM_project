# %%
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from atpbar import atpbar
from mantichora import mantichora
import subprocess

os.chdir('/Users/pita/Documents/Rene/GAM_project/genome_architecture_entropy/')
#os.chdir('/fast/AG_Metzger/rene/GAM_project/genome_architecture_entropy/')
from src.toymodel import chaingen as gen

# import run
from ext.md_soft import run as rn

path_project    = Path.cwd()
path_toymodels  = str(path_project / 'data/toymodels/')
path_restraints = str(path_project / 'data/md_soft/restraints/')
path_out        = str(path_project / 'data/md_soft/')
#path_config     = str(path_project / 'data/md_soft/config.ini')
path_ini_struct = str(path_project / 'data/md_soft/initial_structure.pdb')
path_run_mdsoft = str(path_project / 'ext/md_soft/run.py')


# %%
def init_tads(seeds):
    # initialize tad boundaries +1, -1 of seed locs
    for i in range(seeds.shape[0]):
        if i == 0:
            tads = np.array([[seeds[i]-1, seeds[i]+1]])
        else:
            tad = np.array([seeds[i]-1, seeds[i]+1])
            tads = np.vstack((tads, tad))

    return tads


def write_tads_to_file(tads, num):
    # convert to list and print to spec
    restraints = tads.tolist()
    file_restraint = path_restraints + '/restraint%d.rst' % (num+1)

    with open(file_restraint, 'w') as f:
        for boundary in restraints:
            boundary.insert(0, ':')
            boundary.insert(2, ' :')
            #tad.append(' >>')
            print(*boundary, sep='', end='\n', file=f)
            #print(*boundary, sep='')

    return None


def generate_boundary_sequence(tads, steps, lbound, rbound):
    iterations = 0
    n_tads = tads.shape[0]
    flag = np.zeros(shape=n_tads)

    # need to check iteratively due to different types (int, None)
    for iteration in steps:
        l = len(iteration)
        max_len = max(l, len(iteration))

    #outer loop: simultaneous restraint changes; inner loop: individual, boundary conditions
    for iteration in steps:
        # remove [0,0] from tads
        tads = tads[tads[:,0] != 0]
        write_tads_to_file(tads, iterations)
        iterations += 1

        for tad, val in enumerate(iteration):
            # no new restraint
            if val == None:
                continue
            
            # remove restraint
            if val == -1:
                tads[tad] = [0, 0]
                flag[tad] = 1
            
            # stop if restraint is already removed
            if tads[tad].all() == 0:
                continue

            left_space = False
            right_space = False
            dist_forw = False
            dist_back = False

            # check spaces depending on position (only, first, last, any)
            if tads.shape[0] == 1:
                left_space = tads[tad][0] > lbound
                right_space = tads[tad][1] < rbound
            elif tad == 0:
                left_space = tads[tad][0] > lbound
                dist_forw = (tads[tad+1][0] - tads[tad][1]) > 1
            elif tad == max_len:
                right_space = tads[tad][1] < rbound
                dist_back = (tads[tad][0] - tads[tad-1][1]) > 1
            else:
                dist_forw = tads[tad+1][0] - tads[tad][1] > 1
                dist_back = tads[tad][0] - tads[tad-1][1] > 1

            # check space requirements and grow tads
            if left_space and dist_forw:
                tads[tad][0] = tads[tad][0] - 1
                tads[tad][1] = tads[tad][1] + 1

            elif right_space and dist_back:
                tads[tad][0] = tads[tad][0] - 1
                tads[tad][1] = tads[tad][1] + 1

            elif dist_forw and dist_back:
                tads[tad][0] = tads[tad][0] - 1
                tads[tad][1] = tads[tad][1] + 1

            elif left_space and right_space:
                tads[tad][0] = tads[tad][0] - 1
                tads[tad][1] = tads[tad][1] + 1

            elif left_space and not dist_forw:
                tads[tad][0] = tads[tad][0] - 1

            elif right_space and not dist_back:
                tads[tad][1] = tads[tad][1] + 1

            elif not left_space and dist_forw:
                tads[tad][1] = tads[tad][1] + 1

            elif not right_space and dist_back:
                tads[tad][0] = tads[tad][0] - 1

            else: 
                flag[tad] = 1

            if flag.all():
                break

    return iterations

        
def run_sim(series_len, chain_len, name):
    colspecs = [(0, 6), (6, 11), (12, 16), (16, 17), (17, 20), (21, 22), (22, 26), (26, 27), (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (76, 78), (78, 80)]
    cols = ['x','y']
    names = ['ATOM', 'serial', 'name', 'altloc', 'resname', 'chainid', 'resseq', 'icode', 'x', 'y', 'z', 'occupancy', 'tempfactor', 'element', 'charge']

    path_config = path_out + '/config_%s.ini' % (name)
    #cmd = 'python ' + path_run_mdsoft + ' -c ' + path_config
    series = np.empty(shape=(series_len, chain_len, 2))

    for state in range(series_len):
        # for state 0 the initial structure is chosen. After that the new minimized structures are used.
        if state == 0: 
            pdb_path = path_out + '/initial_structure_%s.pdb' % (name)  
            skiprows = None
        else: 
            # replace restraints and input structure paths in config file
            with open(path_out + '/config.ini', 'r') as config_file:
                list_of_lines = config_file.readlines()
            list_of_lines[2] = 'INITIAL_STRUCTURE_PATH =' + pdb_path + '\n'
            #list_of_lines[3] = 'FORCEFIELD_PATH =' + path_run_mdsoft + '/forcefields/classic_sm_ff.xml' + '\n'
            list_of_lines[18] = 'HR_RESTRAINTS_PATH =' + path_restraints + '/restraint%d.rst\n' % (state)
            list_of_lines[24] = 'MINIMIZED_FILE =' + path_out + '/min_struct_%s.pdb' % (name) + '\n'
            with open(path_config, 'w') as config_file:
                config_file.writelines(list_of_lines)

            # run md_soft with args: python run.py -c config.ini
            #subprocess.run(cmd, shell=True)#, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            rn.main([path_run_mdsoft, '-c', path_config])

            pdb_path = path_out + '/min_struct_%s.pdb' % (name)
            skiprows = [0]

        pdb = pd.read_fwf(pdb_path, names=names, skiprows=skiprows, colspecs=colspecs, usecols=lambda x: x  in cols).dropna()
        series[state,:,:] = np.array(pdb, dtype=float)

    return series


if __name__ == '__main__': 
    # toymodel1
    #tad_seeds = np.array([7, 20, 31]) 

    # toymodel2: different loop extrusion rates
    #tad_seeds = np.array([10, 20, 28]) 
    #steps = []
    #for loci in range(1, chain_len + 1):
    #    if loci % 2 == 0:
    #        steps.append(1)
    #    if loci % 3 == 0:
    #        steps.append(2)
    #    steps.append(0)

    # toymodel3: different extrusion start times
    #tad_seeds = np.array([5, 25])
    #steps = [[0],[0],[0],[0],[0],[0],[0,1],[0,1],[0,1],[0,1],[0,1],[1],[1],[1],[1],[1],[1]]

    # toymodel4_2: 1 longer loop with a unique realisation per slice
    #chain_len = 31
    #tad_seeds = np.array([15])
    #steps = [[0] for i in range(15)]

    # toymodel5: 2 loops with variable number of realisations per time step
    #chain_len = 31
    #tad_seeds = np.array([8, 23])
    #steps = [[0],[0,1],[0],[0,1],[0],[0,1],[None],[1],[None],[1],[None],[1],[None]]

    # toymodel6: 2 loops 
    model_n = 6
    chain_len = 25
    tad_seeds = np.array([7, 15])
    steps = [[0],[0,1],[0,1],[-1,1],[1],[1],[1],[1]]

    # toymodel7: longer dynamic model with cohesin dissociation
    #chain_len = 50
    tad_seeds = np.array([10,18,28,35])
    steps = [[0,1,2,3], [0,1,2,3], [0,-1,2,3], [0,-1,3], [0,3], [0,3], [0,3]]


    n_realisations = 500

    series = np.zeros(shape=(n_realisations, len(steps), chain_len, 2))
    for itr in tqdm(range(n_realisations)):
        # initialise coordinates to pdb input file 
        coords = gen.self_avoiding_random_walk(chain_len)
        file_name = '/initial_structure_task_0.pdb'
        pdbfile = path_out + file_name
        gen.save_points_as_pdb(coords, pdbfile)

        # start model generation
        tads_initial = init_tads(tad_seeds)
        timesteps = generate_boundary_sequence(tads_initial, steps, lbound=1, rbound=chain_len)
        series[itr] = run_sim(timesteps, chain_len, 'task_0')

    file = '/model%d/toymodel%d_multi_%d.npy' % (model_n, model_n, n_realisations)
    np.save(path_toymodels + file, series)


    #n_cores = 2
    #tads_initial = init_tads(tad_seeds)
    #timesteps = generate_boundary_sequence(tads_initial, steps, lbound=1, rbound=chain_len)
    #def task(name, realisations):
    #    series = np.zeros(shape=(realisations, len(steps), chain_len, 2))
    #    for itr in atpbar(range(realisations), name=name):
    #        # initialise coordinates to pdb input file 
    #        coords = gen.self_avoiding_random_walk(chain_len)
    #        file_name = '/initial_structure_%s.pdb' % (name)
    #        pdbfile = path_out + file_name
    #        gen.save_points_as_pdb(coords, pdbfile)
    #        series[itr] = run_sim(timesteps, chain_len, name)
    #    return series

    #with mantichora(nworkers=n_cores) as mcore:
    #    for i in range(n_cores):
    #        mcore.run(task, 'task_%d' % i, n_realisations // n_cores)
    #    returns = mcore.returns()
    #series = np.concatenate([returns[i] for i in range(n_cores)], axis=0)