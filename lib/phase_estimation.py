


def create_hamiltonians(list_of_spins, list_of_params, mol_params):
    from lib.qpe import QPE
    from lib.cr2dataset import get_pot_cr2, hartree
     # Dictionary to store the Hamiltonians for each spin and parameter set
    hamiltonians = {}

    for spin, params in zip(list_of_spins, list_of_params):
        # Copy molecular parameters and update the name
        mol_params = mol_params.copy()
        mol_params['name'] += f'_{spin}'
        
        # Set DVR options based on the current parameter set
        dvr_options = {
            'type': '1d',
            'box_lims': (params[0], params[1]),
            'dx': (params[1] - params[0]) / 32,
            'count': 32
        }
        
        # Obtain the potential for the molecule at the given spin
        pot, lims = get_pot_cr2(spin)
        
        # Create the QPE object and obtain the Hamiltonian
        qpe = QPE(mol_params, pot)
        h_dvr = qpe.get_h_dvr(dvr_options, J=0) * hartree  
        
        # Store the Hamiltonian in the dictionary
        hamiltonians[(spin, tuple(params))] = h_dvr

    return hamiltonians
    
def convert_to_unitary(matrix, n, wire_order, time=1.0):
    import pennylane as qml
    unitary = qml.ApproxTimeEvolution(matrix, time, n)
    U_matrix = qml.matrix(unitary,wire_order)
    return U_matrix


def find_true_phases(U):
   import numpy as np
   eigvals, eigvecs = np.linalg.eig(U)
   phi_ind = 0
   truephi = np.angle(eigvals[phi_ind])
   return truephi


    
    
    