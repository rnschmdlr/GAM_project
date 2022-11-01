# %%
import os
import numpy as np
import pandas as pd
import subprocess
from tqdm.auto import tqdm
from pathlib import Path
from Bio.PDB import PDBIO
from Bio.PDB import Selection
from Bio.PDB.PDBParser import PDBParser

os.chdir('/Users/pita/Documents/Rene/GAM_project/genome_architecture_entropy/src/')
#os.chdir('/fast/AG_Metzger/rene/GAM_project/genome_architecture_entropy/src/')
from toymodel import chaingen as gen

path_cwd        = Path.cwd()
path_project    = path_cwd.parents[0]
path_restraints = str(path_project / 'data/md_soft/restraints/')
path_out        = str(path_project / 'data/md_soft/')
path_config     = str(path_project / 'data/md_soft/config.ini')
path_ini_struct = str(path_project / 'data/md_soft/initial_structure.pdb')
path_run_mdsoft = str(path_project / 'ext/md_soft/run.py')



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
    file_restraint = path_restraints + '/restraint%d.rst' % (num)

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


def translocate_xy(pdb_input, pdb_output):
    # read unchanged file to memory
    with open(pdb_input, 'r') as pdb_input_file:
        list_of_lines = pdb_input_file.readlines()

    # load structures and lists of atoms
    parser = PDBParser()
    structure_init = parser.get_structure('init', pdb_input)
    structure_out = parser.get_structure('out', pdb_output)
    # TODO check if length of chains matches

    atoms_init = Selection.unfold_entities(structure_init, 'A')
    atoms_out = Selection.unfold_entities(structure_out, 'A')

    # replace coordinates of atoms_init with those of atoms_out
    for atom in range(len(atoms_out)):
        coord_out = atoms_out[atom].get_coord()
        atoms_init[atom].set_coord(coord_out)

        if coord_out[2] > 1:
            print('Warning: z-coordinate > 1 for atom' + str(atom+1) + coord_out)
        
    io = PDBIO()
    io.set_structure(structure_init)
    io.save(pdb_input)

    with open(pdb_input, 'a') as pdb_input_file_edited:
        pdb_input_file_edited.writelines(list_of_lines[36:])

        
def run_sim(series_len, chain_len, pdb_init):
    colspecs = [(0, 6), (6, 11), (12, 16), (16, 17), (17, 20), (21, 22), (22, 26),
                    (26, 27), (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (76, 78), (78, 80)]
    cols = ['x','y']
    names = ['ATOM', 'serial', 'name', 'altloc', 'resname', 'chainid', 'resseq',
                'icode', 'x', 'y', 'z', 'occupancy', 'tempfactor', 'element', 'charge']
    path_min_pdb = path_out + '/min_struct.pdb'
    cmd = 'python ' + path_run_mdsoft + ' -c ' + path_config
    series = np.empty((series_len+1, chain_len, 2))

    for state in range(series_len):
        # replace restraints and input structure paths in config file
        with open(path_config, 'r') as config_file:
            list_of_lines = config_file.readlines()
        list_of_lines[18] = 'HR_RESTRAINTS_PATH =' + path_restraints + '/restraint%d.rst\n' % (state)
        # for state 0 the initial structure is chosen. After that the new minimized structures are used.
        if state > 0:
            list_of_lines[2] = 'INITIAL_STRUCTURE_PATH =' + path_min_pdb + '\n'
        else:
            list_of_lines[2] = 'INITIAL_STRUCTURE_PATH =' + pdb_init + '\n'
        with open(path_config, 'w') as config_file:
            config_file.writelines(list_of_lines)

        # run md_soft with args: python run.py -c config.ini
        #subprocess.run(cmd, shell=True)

        # execute subprocess.run with cmd and non verbose
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # this adds the initial structure as T=0 to the coordinate array first
        #if state == 0:
        #    pdb = pd.read_fwf(pdb_init, names=names, colspecs=colspecs, usecols=lambda x: x  in cols).dropna()
        #    series[state,:,:] = np.array(pdb, dtype=float)

        pdb = pd.read_fwf(path_min_pdb, names=names, skiprows=[0], colspecs=colspecs, usecols=lambda x: x  in cols).dropna()
        series[state+1,:,:] = np.array(pdb, dtype=float)

    return series


if __name__ == '__main__': 
    # toymodel6: 2 loops 
    chain_len = 25
    tad_seeds = np.array([7, 15])
    steps = [[0],[0,1],[0,1],[-1,1],[1],[1],[1],[1]]

    # toymodel4_2: 1 longer loop with a unique realisation per slice
    #chain_len = 31
    #tad_seeds = np.array([15])
    #steps = [[0] for i in range(15)]

    # toymodel5: 2 loops with variable number of realisations per time step
    #chain_len = 31
    #tad_seeds = np.array([8, 23])
    #steps = [[0],[0,1],[0],[0,1],[0],[0,1],[None],[1],[None],[1],[None],[1],[None]]

    realisations = 500
    series = np.zeros(shape=(realisations, len(steps)+1, chain_len, 2))
    for itr in tqdm(range(realisations)):
        # initialise coordinates to pdb input file 
        coords = gen.self_avoiding_random_walk(chain_len)
        file_name = '/initial_structure.pdb'
        pdbfile = str(path_out + file_name)
        gen.save_points_as_pdb(coords, pdbfile)

        # start model generation
        tads_initial = init_tads(tad_seeds)
        timesteps = generate_boundary_sequence(tads_initial, steps, lbound=1, rbound=chain_len)
        series[itr] = run_sim(timesteps, chain_len, pdbfile)

    series_grouped = np.reshape(series, newshape=(realisations * (timesteps+1), chain_len, 2), order='F')
    series_indiv = np.reshape(series, newshape=(realisations * (timesteps+1), chain_len, 2), order='C')
    model = '/toymodel6_multi_1000.npy'
    np.save(path_out + model, series)

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
