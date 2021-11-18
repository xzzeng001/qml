# for parallel
import multiprocessing as mp
num_proc=mp.cpu_count()
import os
os.environ["OMP_NUM_THREADS"] = str(num_proc+1)

#import os
from user_config import api_token
from qiskit import IBMQ # type: ignore
from pyscf_func import obtain_Hamiltonian
import numpy as np
import save_data as sd
import sys

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from execution import *

if __name__ == "__main__":
   

    simulator=True
##    # if use the real quantum computor
##    if not simulator: 
##       # Load IBMQ account
##       try:
##           if api_token != '':
##               IBMQ.enable_account(api_token)
##           else:
##               IBMQ.load_account()
##       except:
##           logger.error('Set your api_token in user_config.py or use IBMQ.save_account(your_api_token) on your machine once.')
                    
    # obtain the Hamiltonian from the specific system
    np.set_printoptions(threshold=np.inf)
  
    draw_circ=True 
    for dist in np.arange(0.7,0.6,-0.1):
        # specify the geometry strcutres:
#        geometry = [["H", [0.0, 0.0, -1.0 * dist]],
#                      ["H", [0.0, 0.0,  1.0 * dist]],
#                      ["Be",[0.0, 0.0,  0.0       ]]]

##        char=["H"]
##        flist=[0.0]
##        rdist=[i for i in np.arange(0.0,37.5,0.75)]
##        geometry=[[char[0*i],[flist[0*i],flist[0*i],rdist[i]]] for i in range(50)]
##        print('geometry:',geometry)
##        sys.exit(0)

        geometry = [["H", [0.0, 0.0, 0.0]],
                    ["H", [0.0, 0.0, 1.0*dist]],
                    ["H", [0.0, 0.0, 2.0*dist]],
                    ["H", [0.0, 0.0, 3.0*dist]]]
        basis  = "sto3g"
        spin   = 0
        charge = 0

        energy, n_qubits, n_orb, n_orb_occ, occ_indices_spin, nterms, ep_cont, ham = obtain_Hamiltonian(geometry,basis,spin,charge,dist,with_fci=True, BK_reduce=True) 

        if n_qubits >= 10:
            draw_circ=False

        noise_frac=1e-3
        ## Choose ansatz for your training run
        ep_final=iteration_optimization(
            simulator=simulator,
            ansatz=Ansatz_Pool(name='dqnn',qnn_arch=[n_qubits,2,n_qubits], measurement_method='swap_trick',hamiltonian=ham,fci_e=energy, nterms=nterms,ep_cont=ep_cont), #,gate_error_probabilities={'u3': [noise_frac,1],'rxx': [noise_frac,2],'ryy': [noise_frac,2],'rzz': [noise_frac,2]}),
#            ansatz=Ansatz_Pool(name='ucc', excited_ranks='gsd', n_qubits=n_qubits, n_orb=n_orb, n_orb_occ=n_orb_occ, occ_indices_spin=occ_indices_spin,measurement_method='PauliExpectation',hamiltonian=ham,fci_e=energy,nterms=nterms,ep_cont=ep_cont),
#            ansatz=Ansatz_Pool(name='hardware',n_qubits=n_qubits, num_entangle=1, measurement_method='PauliExpectaion',hamiltonian=ham,fci_e=energy, nterms=nterms,ep_cont=ep_cont),
            adapt_vqe=False, # for the adapt vqe
            adapt_tol=1e-3,  # the tolerance of adapt_vqe
            adapt_max_iter=20, # the maximum steps for the adapt_vqe
            shots=1024,    # the maximum iterations
            epochs=10000, # total epochs about 3.5*epochs
            optimize_method="COBYLA",#"L_BFGS_B",#"Nelder-Mead",#"COBYLA",#"Adam",#"Nelder-Mead",#"COBYLA"
            learning_rate=1e-1,
            analy_grad=False, # True for the analytical gradient method, False for Numerical gradient 
            order_of_derivative=2,   # the order of derivative 
            epsilon=0.25,            # the step for the derivative      
            draw_circ=draw_circ,
            load_params_from='./params.txt' if os.path.exists('params.txt') else None,
            device_name="qasm_simulator",#"ibmq_belem",#"qasm_simulator",
            simulation_method="matrix_product_state",
            optimization_level=3)

        ## save the energy
        plot_list_energy=np.array([[dist,energy,ep_final,ep_final-energy]])
        sd.save_and_plot_energy(plot_list_energy)
        if draw_circ:
            draw_circ=False 
