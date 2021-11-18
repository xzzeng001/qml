# for parallel
import multiprocessing as mp
num_proc=mp.cpu_count()
import os
os.environ["OMP_NUM_THREADS"] = str(num_proc+1)

from pyscf_func import obtain_Hamiltonian
from dqnn_utils import generate_canonical_circuit_all_neurons, transform_the_hamiltonian,save_data

import numpy as np
import sys

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.opflow.list_ops import SummedOp
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.quantum_info import DensityMatrix, partial_trace
from qiskit.algorithms.optimizers import ADAM,COBYLA,P_BFGS,NFT,QNSPSA,NELDER_MEAD,L_BFGS_B

from typing import Union, Optional, List, Tuple, Any, Dict

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
   
    # obtain the Hamiltonian from the specific system
    np.set_printoptions(threshold=np.inf)
  
    dist=0.7 
    # specify the geometry strcutres:
#    geometry = [["H", [0.0, 0.0, -1.0 * dist]],
#                  ["H", [0.0, 0.0,  1.0 * dist]],
#                  ["Be",[0.0, 0.0,  0.0       ]]]

##    char=["H"]
##    flist=[0.0]
##    rdist=[i for i in np.arange(0.0,37.5,0.75)]
##    geometry=[[char[0*i],[flist[0*i],flist[0*i],rdist[i]]] for i in range(50)]
##    print('geometry:',geometry)
##    sys.exit(0)

    geometry = [["H", [0.0, 0.0, 0.0]],
                ["H", [0.0, 0.0, 1.0*dist]],
                ["H", [0.0, 0.0, 2.0*dist]],
                ["H", [0.0, 0.0, 3.0*dist]]]
    basis  = "sto3g"
    spin   = 0
    charge = 0

    fci_energy, n_qubits, n_orb, n_orb_occ, occ_indices_spin, nterms, ep_cont, ham = obtain_Hamiltonian(geometry,basis,spin,charge,dist,with_fci=True, BK_reduce=True) 

    qnn_arch=[n_qubits,2,2,n_qubits]
    num_params = sum([qnn_arch[l]*qnn_arch[l+1]*3 + (qnn_arch[l])*3 for l in range(len(qnn_arch)-1)]) + qnn_arch[-1]*3
    params_per_layer = [qnn_arch[l]*qnn_arch[l+1]*3 + (qnn_arch[l])*3 for l in range(len(qnn_arch)-1)]
    params=np.random.uniform(high=2*np.pi, size=(num_params))
   
    print('The Architecture of network: ',qnn_arch) 
    print('THe totall parameters of network:',num_params)
    sys.stdout.flush()

    class cont_value:
        def __init__(self,min_e):
            self.min_e=min_e

    ansatz=cont_value(9999)
    epochs=10000
    plot_list_cost: List[List[Union[float]]] = []

    def calculate_expectation(params):
        input_dm=DensityMatrix.from_instruction(QuantumCircuit(qnn_arch[0]))
        # going through each output layer
        for layer in range(len(qnn_arch)-1):
            input_index=[i for i in range(qnn_arch[layer])]
            tmp_out=DensityMatrix.from_instruction(QuantumCircuit(qnn_arch[layer+1]))
            out_dm=tmp_out.expand(input_dm)

            # the resepctive parameters
            layer_params = params[np.sign(layer)*sum(params_per_layer[:layer]):sum(params_per_layer[:layer+1])]

            # append subcircuit connecting all neurons of (layer+1) to layer
            circ=generate_canonical_circuit_all_neurons(qnn_arch,layer_params, layer=layer+1)

            DM=out_dm.evolve(circ)
            input_dm=partial_trace(DM,input_index)

        # add last U3s to all output qubits (last layer)
        last_params=params[-qnn_arch[-1]*3:]
        circ = QuantumCircuit(qnn_arch[-1]) 
        for i in range(qnn_arch[-1]):
            circ.u(last_params[i*3], last_params[i*3+1], last_params[i*3+2], i)

        out_dm=input_dm.evolve(circ)

        Ham_op=transform_the_hamiltonian(ham,qnn_arch[-1],ep_cont)
        energy=out_dm.expectation_value(Ham_op)
        ep_final=energy.real

        plot_list_cost.append([ep_final,fci_energy,ep_final-fci_energy])
 
        if abs(ep_final-fci_energy) < abs(ansatz.min_e-fci_energy):
            ansatz.min_e=ep_final
            save_data(all_params_epochs=params, plot_list_cost=plot_list_cost)
        else:
            save_data(plot_list_cost=plot_list_cost)

        return ep_final

    COBYLA(maxiter=epochs).optimize(num_vars=len(params),objective_function=calculate_expectation,initial_point=params)

    

