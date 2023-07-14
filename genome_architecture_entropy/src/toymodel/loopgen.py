# %% Imports and function definitions
'''Imports and function definitions'''

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

os.chdir('/Users/pita/Documents/Rene/GAM_project/genome_architecture_entropy/')
#os.chdir('/fast/AG_Metzger/rene/GAM_project/genome_architecture_entropy/')
from src.toymodel import chaingen as gen
from ext.md_soft import run as rn
os.chdir('/Users/pita/Documents/Rene/GAM_project/genome_architecture_entropy/')

path_project    = Path.cwd()
path_toymodels  = str(path_project / 'data/toymodels/')
path_restraints = str(path_project / 'data/md_soft/restraints/')
path_out        = str(path_project / 'data/md_soft/')
path_ini_struct = str(path_project / 'data/md_soft/initial_structure.pdb')
path_run_mdsoft = str(path_project / 'ext/md_soft/run.py')


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def init_tads(seeds):
    # initialize tad boundaries +1, -1 of seed locs
    for i in range(seeds.shape[0]):
        if i == 0:
            tads = np.array([[seeds[i]-1, seeds[i]+1]])
        else:
            tad = np.array([seeds[i]-1, seeds[i]+1])
            tads = np.vstack((tads, tad))

    return tads


def write_tads_to_file(tads, num=0, dist=''):
    # convert to list and print to spec
    restraints = tads.tolist()
    file_restraint = path_restraints + '/restraint%d.rst' % (num+1)

    with open(file_restraint, 'w') as f:
        for boundary in restraints:
            boundary.insert(0, ':')
            boundary.insert(2, ' :')
            boundary.append(' ' + str(dist))
            print(*boundary, sep='', end='\n', file=f)
            #print(*boundary, sep='')

    return None


def generate_boundary_sequence(seeds, steps, lbound, rbound):
    iterations = 0
    n_tads = seeds.shape[0]
    flag_stop = np.zeros(shape=n_tads)
    flag_start = np.zeros(shape=n_tads)
    tads = np.zeros((n_tads, 2), dtype=int)

    #outer loop: simultaneous constraint changes; inner loop: individual, boundary conditions
    for iteration in steps:
        #print(tads)
        for tad, val in enumerate(iteration):
            #print(f"tad: {tad}, val: {val}")

            # no new constraint
            if val == None or val == 0:
                continue
            
            # add first constraint for a loop
            elif flag_start[tad] == 0:
                tads[tad] = np.array([init_tads(seeds)[tad]])
                flag_start[tad] = 1
                #print('constraint added')

            # stop if constraint is already removed
            elif flag_stop[tad] == 1:
                continue

            # remove constraint
            elif val == -1:
                tads[tad] = [0,0]
                #max_len -= 1
                flag_stop[tad] = 1

            # grow tads
            else:
                #print('looking for space')
                # reset spacing
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
                    dist_forw = (tads[tad+1][0] - tads[tad][1]) > val
                elif tad == n_tads-1: # last tad
                    right_space = tads[tad][1] < rbound
                    dist_back = (tads[tad][0] - tads[tad-1][1]) > val
                else:
                    dist_forw = tads[tad+1][0] - tads[tad][1] > val
                    dist_back = tads[tad][0] - tads[tad-1][1] > val

                # check space requirements and grow tads
                if left_space and dist_forw:
                    tads[tad][0] = tads[tad][0] - val
                    tads[tad][1] = tads[tad][1] + val

                elif right_space and dist_back:
                    tads[tad][0] = tads[tad][0] - val
                    tads[tad][1] = tads[tad][1] + val

                elif dist_forw and dist_back:
                    tads[tad][0] = tads[tad][0] - val
                    tads[tad][1] = tads[tad][1] + val

                elif left_space and right_space:
                    tads[tad][0] = tads[tad][0] - val
                    tads[tad][1] = tads[tad][1] + val

                elif left_space and not dist_forw:
                    tads[tad][0] = tads[tad][0] - val

                elif right_space and not dist_back:
                    tads[tad][1] = tads[tad][1] + val

                elif not left_space and dist_forw:
                    tads[tad][1] = tads[tad][1] + val

                elif not right_space and dist_back:
                    tads[tad][0] = tads[tad][0] - val

                else: 
                    flag_stop[tad] = 1

                if flag_stop.all():
                    break

        tads_keep = tads[tads[:,0] != 0]
        if iterations >= 3 and iterations < 5:
            # add constraint 0, 45 to tads_keep
            tads_keep = np.vstack((tads_keep, np.array([15,54])))
        if iterations >= 4 and iterations < 5:
            # add constraint 0, 45 to tads_keep
            tads_keep = np.vstack((tads_keep, np.array([65,85])))
        if iterations >= 6:
            tads_keep = np.vstack((tads_keep, np.array([10,90])))
        write_tads_to_file(tads_keep, iterations)
        iterations += 1

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
            with HiddenPrints():
                rn.main([path_run_mdsoft, '-c', path_config])

            pdb_path = path_out + '/min_struct_%s.pdb' % (name)
            skiprows = [0]

        pdb = pd.read_fwf(pdb_path, names=names, skiprows=skiprows, colspecs=colspecs, usecols=lambda x: x  in cols).dropna()
        series[state,:,:] = np.array(pdb, dtype=float)

    return series



# %% Compute loop extrusion process
'''Compute loop extrusion process'''

# set parameters
# toymodel1
#tad_seeds = np.array([7, 20, 31]) 

# toymodel2: different loop extrusion rates
#tad_seeds = np.array([10, 20, 28]) 
#model_n = 2
#steps = []
#chain_len = 36
#steps = [[0], [1,0], [2,0], [1,0], [0,1,2], [0], [0,1], [0,2]]

# toymodel3: different extrusion start times
#tad_seeds = np.array([5, 25])
#steps = [[0],[0],[0],[0],[0],[0],[0,1],[0,1],[0,1],[0,1],[0,1],[1],[1],[1],[1],[1],[1]]

# toymodel4_3: 1 longer loop with a unique realisation per slice
#chain_len = 15
#tad_seeds = np.array([8])
#steps = [[1] for i in range(6)]

# toymodel5: 2 loops with variable number of realisations per time step
#chain_len = 31
#tad_seeds = np.array([8, 23])
#steps = [[0],[0,1],[0],[0,1],[0],[0,1],[None],[1],[None],[1],[None],[1],[None]]

# toymodel6: 2 loops 
#model_n = 6
#chain_len = 25
#tad_seeds = np.array([7, 15])
#steps = [[0],[0,1],[0,1],[-1,1],[1],[1],[1],[1]]

# toymodel7: longer dynamic model with cohesin dissociation
#model_n = 7
#chain_len = 70
#tad_seeds = np.array([12,24,32,44,53])
#steps = [[2,2,2,2,2], [2,2,2,2,2], [2,2,-1,2,2], [2,2,None,2,2], [2,2,None,2,2], [2,2,None,2,2], [2,2,None,2,2]]

# toymodel 4_3: figure 1 in thesis
#model_n = 9
#chain_len = 25
#tad_seeds = np.array([7,12,21])
#steps = [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]

# toymodel 9: 2 loops that appear one after the other
#model_n = 9
#chain_len = 28
#tad_seeds = np.array([8,21])
#steps = [[1,None],[1,None], [1,None], [1,1], [1,1], [None,1], [None,1], [None,1]]

# toymodel 10: TAD formation over 10 time steps with 5 subtads
model_n = 10
chain_len = 100
tad_seeds = np.array([34,41,48,64,74])
steps = [[1, None, None, 1, None],[1, None, None, 1, 1],[1,1,1,1,1],[1,1,1,1,1],[0,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0]]
# nach timestep 3 subTAD 1 verschmilzt aus loop 1-3
# nach timestep 4 subTAD 2 verschmilzt aus loop 4-5
# nach timestep 6 verschmlezen subTAD 1 und 2 zu einem TAD

# start computation
n_realisations = 250
series = np.zeros(shape=(n_realisations, len(steps)+1, chain_len, 2)) # +2 to include zeroeth and initital structures

for itr in tqdm(range(n_realisations)):
    # initialise coordinates to pdb input file 
    coords = gen.self_avoiding_random_walk(chain_len)
    file_name = '/initial_structure_task_0.pdb'
    pdbfile = path_out + file_name
    gen.save_points_as_pdb(coords, pdbfile)

    # start model generation
    n_timesteps = generate_boundary_sequence(tad_seeds, steps, lbound=1, rbound=chain_len-1)
    series[itr] = run_sim(n_timesteps+1, chain_len, 'task_0') # +1 to compute initital structure

file = '/model%d/toymodel%d_multi_%d_4.npy' % (model_n, model_n, n_realisations)
np.save(path_toymodels + file, series)

# toymodel8: A Only closed conformation. B Only open conformation. C Mix of open and closed conformations. D Average of open and closed conformations. 
# model A
#model_n = 10
#chain_len = 36
#n_realisations = 100
#steps = 2
#series = np.zeros(shape=(n_realisations, chain_len, 2))
#for itr in tqdm(range(n_realisations)):
#    coords = gen.self_avoiding_random_walk(chain_len)
#    file_name = '/initial_structure_model_A.pdb'
#    pdbfile = path_out + file_name
#    gen.save_points_as_pdb(coords, pdbfile)
#    tads_initial = np.array([[17,22]])
#    write_tads_to_file(tads_initial, 0, '0.2 15000')
#    series[itr] = run_sim(steps, chain_len, 'model_A')[1,:,:]
# concatenate series1 and series2 and series3 to one array
#series = np.concatenate((series1, series2, series3), axis=0)
    

# %% '''Multicore processing'''
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

# %% Quick coordinate viewer
'''Quick coordinate viewer'''

from matplotlib import pyplot as plt
#%matplotlib inline
plt.rcParams['figure.dpi'] = 150

for realisation in range(min(5, n_realisations)):
    for state in range(1, n_timesteps+1):
        x = series[realisation].T[0,:,state]
        y = series[realisation].T[1,:,state]
        xmin = np.min(x)
        xmax = np.max(x)
        width = (xmax - xmin)
        ymin = np.min(y)
        ymax = np.max(y)
        height = (ymax - ymin)

        ex_w = 0.33
        ex_h = 0.33

        plt.rcParams["figure.figsize"] = (width*ex_w, height*ex_h)
        plt.plot(x, y, '-k', linewidth=1.8,  zorder=1)
        plt.scatter(x, y, color='black', s=50, zorder=2)
        plt.tick_params(labelleft=False, labelbottom=False)
        #plt.axis('off')
        
        plt.xlim(xmin-ex_w*width, xmax+ex_w*width)
        plt.ylim(ymin-ex_h*height, ymax+ex_h*height)

        plt.show()
    plt.clf()

