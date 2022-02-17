# %%
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path

path_cwd        = Path.cwd()
path_restraints = path_cwd / 'md_soft/restraints/'
path_out        = path_cwd / 'md_soft/out/'
path_config     = path_cwd / 'md_soft/config.ini'
path_ini_struct = path_cwd / 'md_soft/initial_structure.pdb'

path_run_mdsoft = path_cwd.parents[2] / 'md_soft/run.py'



def init_tads(seeds):
    # initialize tad boundaries +1, -1 of seed locs
    for i in range(seeds.shape[0]):
        if i == 0:
            tads = np.array([seeds[i]-1, seeds[i]+1])
        else:
            tad = np.array([seeds[i]-1, seeds[i]+1])
            tads = np.vstack((tads, tad))

    return(tads)


def write_tads_to_file(tads, num, prnt=False):
    # convert to list and print to spec
    restraints = tads.tolist()
    file_restraint = str(path_restraints) + '/restraint%d.rst' % (num)

    with open(file_restraint, 'w') as f:
        for tad in restraints:
            tad.insert(0, ':')
            tad.insert(2, ' :')
            #tad.append(' >>')
            print(*tad, sep='', end='\n', file=f)
            if prnt == True: print(*tad, sep='')

    return None


def generate_boundary_sequence(tads, lbound=1, rbound=35):
    flag = 0 #const
    cond = True
    iterations = 0

    # move boundaries by 1 in either direction. outer loop: ensemble of tads
    while cond: 
        # breaks when no boundaries were moved, e.g. all flags are true
        if flag == tads.shape[0]:
            cond = False
        flag = 0 # reset flag

        write_tads_to_file(tads, iterations, prnt=False)

        #inner loop: each tad, boundary conditions
        for tad in range(tads.shape[0]):
            left_space = False
            right_space = False
            dist_forw = False
            dist_back = False

            # check spaces depending on position (first, last, any)
            if tad == 0:
                left_space = tads[tad][0] > lbound
                dist_forw = (tads[tad+1][0] - tads[tad][1]) > 2
            elif tad == tads.shape[0]-1:
                right_space = tads[tad][1] < rbound
                dist_back = (tads[tad][0] - tads[tad-1][1]) > 2
            else:
                dist_forw = tads[tad+1][0] - tads[tad][1] > 2
                dist_back = tads[tad][0] - tads[tad-1][1] > 2

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

            elif left_space and not dist_forw:
                tads[tad][0] = tads[tad][0] - 1

            elif right_space and not dist_back:
                tads[tad][1] = tads[tad][1] + 1

            elif not left_space and dist_forw:
                tads[tad][1] = tads[tad][1] + 1

            elif not right_space and dist_back:
                tads[tad][0] = tads[tad][0] - 1

            else: 
                flag = flag + 1        
        
        iterations += 1

    return iterations


def run_sim(series_len, chain_len):
    # run md_soft with args: python run.py -c config.ini
    cmd = ['python', str(path_run_mdsoft), '-c', str(path_config)]
    cmd = 'python ' + str(path_run_mdsoft) + ' -c ' + str(path_config)

    global xy_series 
    xy_series = np.empty((series_len, chain_len, 2))

    for state in range(series_len):
        # replace restraints and input structure paths in config file
        with open(path_config, 'r') as config_file:
            list_of_lines = config_file.readlines()
            
        list_of_lines[18] = 'HR_RESTRAINTS_PATH =' + str(path_restraints) + '/restraint%d.rst\n' % (state)
        list_of_lines[2] = 'INITIAL_STRUCTURE_PATH =' + str(path_ini_struct) + '\n'

        if state == 1:
            list_of_lines[2] = 'INITIAL_STRUCTURE_PATH =' + str(path_out) + '/min_struct.pdb\n'

        with open(path_config, 'w') as config_file:
            config_file.writelines(list_of_lines)

        # extract xy coordinates
        if state > -1:
            subprocess.run(cmd, shell=True)

            # remove cols of no interest for numpy array creation
            colspecs = [(0, 6), (6, 11), (12, 16), (16, 17), (17, 20), (21, 22), (22, 26),
                        (26, 27), (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (76, 78),
                        (78, 80)]
            cols = ['x','y']
            names = ['ATOM', 'serial', 'name', 'altloc', 'resname', 'chainid', 'resseq',
                    'icode', 'x', 'y', 'z', 'occupancy', 'tempfactor', 'element', 'charge']
            file = path_out / 'min_struct.pdb'
            pdb = pd.read_fwf(file, names=names, skiprows=[0], colspecs=colspecs, usecols=lambda x: x  in cols).dropna()
            
            # append to 3d numpy array
            xy_series[state,:,:] = np.array(pdb, dtype=float)

            # parse into new initial structure


    return xy_series


if __name__ == '__main__': 
    # seed:
    #    |            |   |   |
    # initialize boundaries
    #    ||          ||  ||   ||
    # generation
    #   | |        |  | |  | |   |
    # |     |    |     ||   ||    |
    #|        ||       ||   ||    |
    # stop when no ops possible

    tad_seeds = np.array([5, 12, 21, 29, 32]) 
    chain_len = 36 # need to change first_initial_structure.pdb first!
    tads_initial = init_tads(tad_seeds)
    series_len = generate_boundary_sequence(tads_initial, rbound=chain_len)
    series_len = 10
    run_sim(series_len, chain_len)

    file = path_out / 'toymodel.npy'
    np.save(file, xy_series)

# %%
