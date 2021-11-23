# for parallel

from pyscf_func import obtain_Hamiltonian
from dqnn_utils import *

import numpy as np
import sys
import os

import torch
import openfermion

from typing import Union, Optional, List, Tuple, Any, Dict

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
   
    # obtain the Hamiltonian from the specific system
    np.set_printoptions(threshold=np.inf)
  
    dist=0.2 
    geometry = [["H", [0.0, 0.0, 0.0]],
                ["H", [0.0, 0.0, 1.0*dist]],
                ["H", [0.0, 0.0, 2.0*dist]],
                ["H", [0.0, 0.0, 3.0*dist]]]
    basis  = "sto3g"
    spin   = 0
    charge = 0

    fci_energy, n_qubits, n_orb, n_orb_occ, occ_indices_spin, nterms, ep_cont, ham = obtain_Hamiltonian(geometry,basis,spin,charge,dist,with_fci=True, BK_reduce=True) 

    qnn_arch=[n_qubits,2,2,2,n_qubits]
    num_params = sum([qnn_arch[l]*qnn_arch[l+1]*3 + (qnn_arch[l])*3 for l in range(len(qnn_arch)-1)]) + qnn_arch[-1]*3
    params_per_layer = [qnn_arch[l]*qnn_arch[l+1]*3 + (qnn_arch[l])*3 for l in range(len(qnn_arch)-1)]
   
    print('The Architecture of network: ',qnn_arch) 
    print('THe total parameters of network:',num_params)
    sys.stdout.flush()

    class cont_value:
        def __init__(self,min_e):
            self.min_e=min_e

    ansatz=cont_value(9999)
    epochs=10000
    plot_list_cost = []
 
    ham_mat = torch.tensor(openfermion.get_sparse_operator(ham).todense(),dtype=torch.float64)

    def calculate_expectation(params):

        input_dm=generate_density_matrix(qnn_arch[0])

        # going through each output layer
        for layer in range(len(qnn_arch)-1):
            tmp_dm=generate_density_matrix(qnn_arch[layer+1])
            out_dm=torch.kron(input_dm.contiguous(),tmp_dm.contiguous())

            # the resepctive parameters
            layer_params = params[np.sign(layer)*sum(params_per_layer[:layer]):sum(params_per_layer[:layer+1])]

            # append subcircuit connecting all neurons of (layer+1) to layer
            circ=generate_canonical_circuit_all_neurons(qnn_arch,layer_params, layer=layer+1)

#            print('circ:',circ.shape)
#            print('out_dm:',out_dm.shape)
            tmp=torch.matmul(circ,out_dm)
            DM=torch.matmul(tmp,circ.T.conj())
            input_dm=partial_trace(DM,[2**qnn_arch[layer],2**qnn_arch[layer+1]])

        # add last U3s to all output qubits (last layer)
        last_params=params[-qnn_arch[-1]*3:]
        circ = add_one_qubit_gates(qnn_arch[-1], 0, last_params)
 
        tmp=torch.matmul(circ,input_dm)
        out_dm=torch.matmul(tmp,circ.T.conj())

#        print('trace(out_dm):',torch.trace(out_dm))
        energy=torch.div(torch.trace(torch.matmul(out_dm,ham_mat)),torch.trace(out_dm))
        ep_final=energy.detach().numpy()

        plot_list_cost=[ep_final,fci_energy,ep_final-fci_energy]
 
        if abs(ep_final-fci_energy) < abs(ansatz.min_e-fci_energy):
            ansatz.min_e=ep_final
            save_data(all_params_epochs=params.detach().numpy(), plot_list_cost=plot_list_cost)
        else:
            save_data(plot_list_cost=plot_list_cost)

        return energy

    if os.path.exists('params.txt'):
        theta=torch.tensor(np.loadtxt('params.txt'),requires_grad=True,dtype=torch.float64)
    else:
        theta=torch.tensor(np.random.uniform(high=2*np.pi,size=(num_params)),requires_grad=True,dtype=torch.float64)
    optimizer = torch.optim.Adam([theta],lr=0.01)
    for n in range(epochs):
        energy=calculate_expectation(theta)
        optimizer.zero_grad()
        energy.backward()
        optimizer.step()
     
