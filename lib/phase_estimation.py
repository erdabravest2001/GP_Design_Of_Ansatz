
import pennylane as qml

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
    unitary = qml.ApproxTimeEvolution(matrix, time, n)
    U_matrix = qml.matrix(unitary,wire_order)
    return U_matrix

def find_true_phases(U):
    import numpy as np
    eigvals, _ = np.linalg.eig(U)
    true_phases = np.angle(eigvals)
    true_phases = np.mod(true_phases, 2*np.pi)
    return true_phases

def init_custom_gates(unitary_matrices):
    custom_gates = {}
    for index, unitary_matrix in enumerate(unitary_matrices, start=1):
        gate_name = f"uffsg{index}"
        custom_gates[gate_name] = qml.QubitUnitary(unitary_matrix, wires=[0, 1, 2, 3, 4])
    return custom_gates

def experiment_probs(k, beta, wires, unitary_matrices, estimation_wires, gate_index):
    import pennylane as qml
    from pennylane import numpy as np
    dev = qml.device('default.qubit', wires=wires)
    custom_gates = init_custom_gates(unitary_matrices)
    @qml.qnode(dev)
    def circuit(k, beta):
        qml.broadcast(qml.Hadamard, wires=range(wires), pattern="single")
        qml.RZ(beta, wires=0)  
        selected_gate = custom_gates[f"uffsg{gate_index}"]
        qml.ctrl(selected_gate, control=estimation_wires)
        qml.broadcast(qml.Hadamard, wires=range(wires), pattern="single")
        return qml.probs(wires=estimation_wires)
    probs = circuit(k, beta)
    return probs



    
    
    