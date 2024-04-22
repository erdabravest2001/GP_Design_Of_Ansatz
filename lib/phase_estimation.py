
import pennylane as qml
import jax.numpy as jnp
from jax import random
import phayes.adaptive

def create_hamiltonians(list_of_spins, list_of_params, mol_params):
    from lib.qpe import QPE
    from lib.cr2dataset import get_pot_cr2, hartree
    hamiltonians = {}
    for spin, params in zip(list_of_spins, list_of_params):
        mol_params = mol_params.copy()
        mol_params['name'] += f'_{spin}'
        dvr_options = {
            'type': '1d',
            'box_lims': (params[0], params[1]),
            'dx': (params[1] - params[0]) / 32,
            'count': 32
        }
        pot, lims = get_pot_cr2(spin)
        qpe = QPE(mol_params, pot)
        h_dvr = qpe.get_h_dvr(dvr_options, J=0) * hartree  
        hamiltonians[(spin, tuple(params))] = h_dvr
    return hamiltonians
    
def convert_to_unitary(matrix, n, wire_order, time=1.0):
    unitary = qml.ApproxTimeEvolution(matrix, time, n)
    U_matrix = qml.matrix(unitary,wire_order)
    return U_matrix

def find_true_phases(U):
    import numpy as np
    eigvals, _ = np.linalg.eig(U)
    phases = np.log(eigvals)/(2 * np.pi * 1j)
    true_phases = phases.real  
    return true_phases

def Rz(beta):
    return jnp.array([[jnp.exp(-1j * beta / 2), 0], [0, jnp.exp(1j * beta / 2)]])

def matrix_power(U, k):
    result = U
    for _ in range(1, k):
        result = jnp.dot(result, U)
    return result

def initial_fourier_state(J):
    from phayes import PhayesState 
    import jax.random as random
    import jax.numpy as jnp
    import numpy as np
    shape= (2, J)
    seed = np.random.randint(0, 2**32 - 1)
    random_key = random.split(random.key(seed))[0]
    return PhayesState(fourier_mode=True, 
                       fourier_coefficients=random.uniform(random_key, shape=shape, minval=0, maxval=2*jnp.pi), 
                       von_mises_parameters=(0.0, 0.0),
                       )
    

    
    

def experiment_probs(k, beta, U, phi_vec, n_qubits_U):
    dim = 2 ** n_qubits_U
    full_dim = 2 * dim
    H = jnp.array([[1, 1], [1, -1]]) / jnp.sqrt(2)
    big_H = jnp.eye(1)
    for _ in range(n_qubits_U + 1):
        big_H = jnp.kron(big_H, H)

    big_Rz = jnp.kron(Rz(beta), jnp.eye(full_dim // 2))

    U_k = matrix_power(U, k)
    controlled_U_k = jnp.block([
        [jnp.eye(full_dim // 2), jnp.zeros((full_dim // 2, full_dim // 2))],
        [jnp.zeros((full_dim // 2, full_dim // 2)), U_k]
    ])
    in_statevector = jnp.kron(jnp.array([1, 0]), phi_vec)
    out_statevector = big_H @ controlled_U_k @ big_Rz @ big_H @ in_statevector
    measurement_probs = jnp.abs(out_statevector) ** 2
    normalized_probs = measurement_probs / jnp.sum(measurement_probs)
    return normalized_probs

def run_experiment(k: int, beta: float, U, phi_vec, n_qubits_U, n_shots: int):
    import numpy as np
    seed = np.random.randint(0, 2**32 - 1)
    random_key = random.split(random.key(seed))[0]
    probs = experiment_probs(k, beta, U, phi_vec, n_qubits_U)
    return random.choice(random_key, a=jnp.arange(len(probs)), p=probs, shape=(len(probs),))


def phase_estimation_iteration(prior, k, U, phi_vec, n_qubits_U, n_shots: int) -> jnp.ndarray:
    from jax import jit
    import phayes
    import numpy as np
    k, beta = jit(phayes.get_k_and_beta)(state=prior, error_rate=0.0, k_max=k)
    likelihood = run_experiment(k, beta, U, phi_vec, n_qubits_U, n_shots)
    m = [5, 6, 7, 8, 9, 10]
    posterior_state = prior[0]*likelihood/np.sum(prior[0]*likelihood)
    return beta, k, posterior_state

def compute_state_probabilities(fourier_coefficients):
    import jax.numpy as jnp
    from jax.numpy.fft import ifft
    state_vector = ifft(fourier_coefficients)
    probabilities = jnp.abs(state_vector)**2
    normalized_probabilities = probabilities / jnp.sum(probabilities)
    return normalized_probabilities

def compute_posterior_mean(probabilities):
    state_indices = jnp.arange(len(probabilities))  
    posterior_mean = jnp.sum(state_indices * probabilities)
    return posterior_mean


def estimate_phase_from_probabilities(probabilities, n_qubits):
    indices = jnp.arange(2**n_qubits)
    phase_index = jnp.sum(indices * probabilities)
    phase = phase_index / (2**n_qubits)
    return phase

