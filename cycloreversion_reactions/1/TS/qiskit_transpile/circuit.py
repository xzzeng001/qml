from qiskit import QuantumCircuit, QuantumRegister, transpile, Aer, execute
from typing import Union, Optional, List, Tuple, Any, Dict
import numpy as np
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
import os

def UCCSD_circuit(num_qubits,num_particles):
    init_state = HartreeFock(num_qubits,num_particles,two_qubit_reduction=False)
    var_form = UCCSD(num_qubits,num_particles, initial_state=init_state,two_qubit_reduction=False)

    num_params=var_form.num_parameters
    if os.path.exists('params.txt'):
        params=np.loadtxt('params.txt')
    else:
        params=np.random.random(num_params)
        
    circ=var_form.construct_circuit(params)

    return num_params, circ

def HAA_circuit(network,ncycle):

    num_qubits=sum(network)
    num_params=2*3*network[0]+ncycle*3*network[0]*network[1]

    if os.path.exists('params.txt'):
        params=np.loadtxt('params.txt')
    else:
        params=np.random.random(num_params)
 
    q_reg = QuantumRegister(num_qubits, 'q')
    # Create a circuit with a register of three qubits
    circ = QuantumCircuit(num_qubits)
    
    iparams = 0

    for i in range(network[0]):
       circ.u(params[iparams], params[iparams+1], params[iparams+2],q_reg[i])
       iparams += 3

    for ii in range(ncycle):
       for i in range(network[1]):
          for j in range(network[0]):
             circ.rxx(2*params[iparams], q_reg[j], q_reg[sum(network[:1])+i])
             circ.ryy(2*params[iparams+1], q_reg[j], q_reg[sum(network[:1])+i])
             circ.rzz(2*params[iparams+2], q_reg[j], q_reg[sum(network[:1])+i])         
             iparams += 3

    for i in range(network[0]):
       circ.u(params[iparams], params[iparams+1], params[iparams+2],q_reg[i])
       iparams += 3

    return num_params, circ

def KMA_circuit(num_qubits,ncycle):

    num_params=3*num_qubits+ncycle*3*num_qubits

    if os.path.exists('params.txt'):
        params=np.loadtxt('params.txt')
    else:
        params=np.random.random(num_params)
 
    q_reg = QuantumRegister(num_qubits, 'q')
    # Create a circuit with a register of three qubits
    circ = QuantumCircuit(num_qubits)
    
    iparams = 0

    for i in range(num_qubits):
       circ.u(params[iparams], params[iparams+1], params[iparams+2],q_reg[i])
       iparams += 3

    for i in range(ncycle):
       for j in range(num_qubits-1):
          circ.cnot(j,j+1)

       for j in range(num_qubits):
          circ.u(params[iparams], params[iparams+1], params[iparams+2],q_reg[j])
          iparams += 3

    return num_params, circ

