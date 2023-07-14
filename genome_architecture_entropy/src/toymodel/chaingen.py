import numpy as np


def random_versor():
    x = 1 #np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    z = 0 #np.random.uniform(-1, 1)
    d = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    return np.array([x / d, y / d, z / d])


def dist(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5  # faster than np.linalg.norm


def self_avoiding_random_walk(n, step=1, bead_radius=0.5, epsilon=0.001):
    points = [np.array([0.01, 0.01, 0])]
    for i in range(n - 1):
        step_is_ok = False
        while not step_is_ok:
            potential_new_step = points[-1] + step * random_versor()
            for j in points:
                d = dist(j, potential_new_step)
                if d < 2 * bead_radius - epsilon:
                    break
            else:
                step_is_ok = True
        points.append(potential_new_step)
    return points


def save_points_as_pdb(points, pdb_file_name, render_connect=True, verbose=False, remarks=' '):
    """Save points in PDB file format."""
    remarks_records = ''
    if remarks:
        for i, remark in enumerate(remarks):
            remarks_records += f'REMARK{i+1:>4} {remark}\n'
    atoms = ''
    n = len(points)
    for i in range(n):
        record_name = 'ATOM'
        serial = i+1
        atom_name = 'B'
        alt_loc = ''
        resid_name = 'BEA'
        chain_id = 'A'
        resid_number = i+1
        iCode = ''
        x = max(points[i][0], -999)
        y = max(points[i][1], -999)
        try:
            z = max(points[i][2], -999)
        except IndexError:
            z = 0.0
        occupancy = 1.0
        tempFactor = 0.0
        element = atom_name
        charge = ''
        atoms += f'{record_name:6}{serial:>5}  {atom_name:3}{alt_loc:1}{resid_name:3} {chain_id}{resid_number:>4}'\
                 f'{iCode:1}   {x:>8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{tempFactor:6.2f}{element:>12} {charge:1}\n'
    terminus = 'TER      '+str(resid_number+1)+'      BEA A  '+str(resid_number)+'\n'
    connects = ''
    if render_connect:
        if n != 1:
            connects = 'CONECT    1    2\n'
            for i in range(2, n):
                connects += 'CONECT{:>5}{:>5}{:>5}\n'.format(i, i - 1, i + 1)
            connects += 'CONECT{:>5}{:>5}\n'.format(n, n - 1)
    pdb_file_content =  atoms + terminus + connects
    with open(pdb_file_name, 'w') as f:
        f.write(pdb_file_content)
    if verbose:
        print("File {} saved...".format(pdb_file_name))
    return pdb_file_name


