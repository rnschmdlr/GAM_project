# %%
import numpy as np
import pandas as pd
import subprocess


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
    with open('/Users/pita/Documents/Rene/Bioinformatics Master/GAM_project/genome_architecture_entropy/toymodel/restraints/restraint%d.rst' % (num), 'w') as f:
        for tad in restraints:
            tad.insert(0, ':')
            tad.insert(2, ' :')
            #tad.append(' H')
            print(*tad, sep='', end='\n', file=f)
            if prnt == True: print(*tad, sep='')

    return 0


def generate_boundary_sequence(tads, lbound=0, rbound=36):
    flag = 0 #const
    cond = True
    iterations = 0

    # move boundaries by 1 in either direction. outer loop: ensemble of tads
    while cond: 
        # breaks when no boundaries were moved, e.g. all flags are true
        if flag == tads.shape[0]:
            cond = False
        flag = 0 # reset flag

        write_tads_to_file(tads, iterations, prnt=True)

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
    cmd = 'python /Users/pita/Documents/Rene/Bioinformatics\ Master/GAM_project/md_soft/md_soft/run.py -c /Users/pita/Documents/Rene/Bioinformatics\ Master/GAM_project/md_soft/md_soft/config.ini'
    global xy_series 
    xy_series = np.empty((series_len, chain_len, 2))

    for state in range(series_len):
        with open('/Users/pita/Documents/Rene/Bioinformatics Master/GAM_project/md_soft/md_soft/config.ini', 'r') as config_file:
            list_of_lines = config_file.readlines()
        list_of_lines[18] = 'HR_RESTRAINTS_PATH = /Users/pita/Documents/Rene/Bioinformatics Master/GAM_project/genome_architecture_entropy/toymodel/restraints/restraint%d.rst\n' % (state)
        with open('/Users/pita/Documents/Rene/Bioinformatics Master/GAM_project/md_soft/md_soft/config.ini', 'w') as config_file:
            config_file.writelines(list_of_lines)

        if state > -1:
            # start md_soft subprocess to start simulation
            proc = subprocess.check_output(cmd, shell=True, universal_newlines=True)
            print(proc)

            #proc = subprocess.check_call(cmd, shell=True)

            # catch output
            filename = '/Users/pita/Documents/Rene/Bioinformatics Master/GAM_project/md_soft/md_soft/example_data/example_result/initial_structure_min.pdb'
            colspecs = [(0, 6), (6, 11), (12, 16), (16, 17), (17, 20), (21, 22), (22, 26),
                        (26, 27), (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (76, 78),
                        (78, 80)]
            cols = ['x','y']
            names = ['ATOM', 'serial', 'name', 'altloc', 'resname', 'chainid', 'resseq',
                    'icode', 'x', 'y', 'z', 'occupancy', 'tempfactor', 'element', 'charge']
            pdb = pd.read_fwf(filename, names=names, skiprows=[0], colspecs=colspecs, usecols=lambda x: x  in cols).dropna()
            
            # append to 3d numpy array
            xy_series[state,:,:] = np.array(pdb, dtype=float)

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

    tad_seeds = np.array([4, 20]) 
    chain_len = 36 # need to change first_initial_structure.pdb first!
    tads_initial = init_tads(tad_seeds)
    series_len = generate_boundary_sequence(tads_initial, rbound=chain_len)
    run_sim(series_len, chain_len)

    # save numpy array
    path = '/Users/pita/Documents/Rene/Bioinformatics Master/GAM_project/genome_architecture_entropy/toymodel'
    np.save(path+'.npy', xy_series)

# %%
