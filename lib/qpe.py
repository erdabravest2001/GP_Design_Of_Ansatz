import pennylane as qml
from pennylane import numpy as np

class QPE:
    def __init__(self, mol_params, pot_fun, log_dir=None) -> None:
        self.mol_params = mol_params
        self.log_dir = log_dir
        self.pot = pot_fun

    def gen_id(self):
        import time
        return str(int(time.time()))

    def get_DVR_Rtheta(self, dvr_options):
        from .ham_gen_arhclmgnh import get_DVR_Rtheta
        from .ham_gen_cr2 import get_dvr_r
        if dvr_options['type'] == 'jacobi':
            Rs_DVR, Xs_K = get_DVR_Rtheta(dvr_options)
            return Rs_DVR, Xs_K
        elif dvr_options['type'] == '1d':
            Rs_DVR = get_dvr_r(dvr_options)
            return Rs_DVR

    def get_h_dvr(self, dvr_options, J=0):
        if dvr_options['type'] == '1d':
            from .ham_gen_cr2 import get_ham_DVR
            return get_ham_DVR(self.pot, dvr_options, mol_params=self.mol_params)
        elif dvr_options['type'] == 'jacobi':
            from .ham_gen_arhclmgnh import get_ham_DVR
            return get_ham_DVR(self.pot, dvr_options, mol_params=self.mol_params)
    
    def excited_cost_function(h_dvr_pauli, ansatz, opt_p_list, betas, vqe, p):
        dev = qml.device('default.qubit', wires=ansatz.num_wires)

        @qml.qnode(dev)
        def get_statevector(params):
            qml.QubitUnitary(ansatz(params), wires=range(ansatz.num_wires))
            return qml.state()

        out = vqe.ExpVal(h_dvr_pauli, p)
        new_state = get_statevector(p)

        for beta, opt_p in zip(betas, opt_p_list):
            state = get_statevector(opt_p)
            # Use qml.math to calculate probabilities
            prob = np.abs(state)**2
            out += beta * prob[0]
        return out
    
    