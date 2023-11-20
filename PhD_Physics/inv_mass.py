import numpy as np

def inv_mass(df:object, particles:list, mass_PDG:list, cut:str='Mbc_1lv > 0'):

    aux = {}; four_vector = {'E': 0, 'px': 0, 'py': 0, 'pz': 0}

    for index, particle in enumerate(particles):

        aux[particle] = {'E': 0,
                         'px': df.query(cut).eval(f"{particle}_px"),
                         'py': df.query(cut).eval(f"{particle}_py"),
                         'pz': df.query(cut).eval(f"{particle}_pz")}

        aux[particle]['E'] = np.sqrt(aux[particle]['px']**2 + aux[particle]['py']**2 + aux[particle]['pz']**2 + mass_PDG[index]**2)

        four_vector['E'] += aux[particle]['E']

        for j in ['px', 'py', 'pz']:
            four_vector[j] += aux[particle][j]

    inv_M = np.sqrt(four_vector['E']**2 - (four_vector['px']**2 + four_vector['py']**2 + four_vector['pz']**2))
    return inv_M

    
def inv_mass_time(df:object, particles:list, mass_PDG:list, time:str='1lv', cut:str='Mbc_1lv > 0'):

    aux = {}; four_vector = {'E': 0, 'px': 0, 'py': 0, 'pz': 0}

    for index, particle in enumerate(particles):

        aux[particle] = {'E': 0,
                         'px': df.query(cut).eval(f"{particle}_px_{time}"),
                         'py': df.query(cut).eval(f"{particle}_py_{time}"),
                         'pz': df.query(cut).eval(f"{particle}_pz_{time}")}

        aux[particle]['E'] = np.sqrt(aux[particle]['px']**2 + aux[particle]['py']**2 + aux[particle]['pz']**2 + mass_PDG[index]**2)

        four_vector['E'] += aux[particle]['E']

        for j in ['px', 'py', 'pz']:
            four_vector[j] += aux[particle][j]

    inv_M = np.sqrt(four_vector['E']**2 - (four_vector['px']**2 + four_vector['py']**2 + four_vector['pz']**2))
    return inv_M
