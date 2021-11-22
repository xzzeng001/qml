# own modules
import sys
import matplotlib.pyplot as plt

import torch

# additional math libs
import numpy as np # type: ignore
from scipy.constants import pi # type: ignore

from typing import Union, Optional, List, Tuple, Any, Dict
import itertools

import logging
logger = logging.getLogger(__name__)


def generate_density_matrix(num_qubits):
    """
    generate the density matrix of zero state
    based on the number of qubits
    """
    input_state=torch.tensor([[1,0]],dtype=torch.complex64)
    out_state=torch.kron(input_state,input_state)
    for ii in range(num_qubits-2):
        out_state=torch.kron(out_state,input_state)
    out=torch.kron(out_state,out_state.T)
    return out
 
def partial_trace(rho, dims, axis=0):
    """
    Takes partial trace over the subsystem defined by 'axis'
    rho: a matrix
    dims: a list containing the dimension of each subsystem
    axis: the index of the subsytem to be traced out
    (We assume that each subsystem is square)
    """
    dims_ = np.array(dims)
    # Reshape the matrix into a tensor with the following shape:
    # [dim_0, dim_1, ..., dim_n, dim_0, dim_1, ..., dim_n]
    # Each subsystem gets one index for its row and another one for its column
    aa=np.concatenate((dims_, dims_), axis=None)
    bb=[]
    for ii in aa:
        bb.append(ii)
    reshaped_rho = rho.reshape(bb)

    # Move the subsystems to be traced towards the end
##    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
##    reshaped_rho = np.moveaxis(reshaped_rho, len(dims)+axis-1, -1)
    reshaped_rho = torch.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = torch.moveaxis(reshaped_rho, len(dims)+axis-1, -1)


    # Trace over the very last row and column indices
#    print('reshaped__rho:',reshaped_rho.shape)
#    sys.exit(0)
#    traced_out_rho1 = np.trace(reshaped_rho.detach().numpy(), axis1=-2, axis2=-1)
    traced_out_rho = torch.diagonal(reshaped_rho, dim1=-2, dim2=-1).sum(dim=2)

#    print('trace_out_rho1:',traced_out_rho1)
#    print('trace_out_rho2:',traced_out_rho2)
#    sys.exit(0)

#    traced_out_rho=torch.from_numpy(traced_out_rho)
    # traced_out_rho is still in the shape of a tensor
    # Reshape back to a matrix
    dims_untraced = np.delete(dims_, axis)
    rho_dim = np.prod(dims_untraced)
##    dims_untraced = torch.delete(dims_, axis)
##    rho_dim = torch.prod(dims_untraced)

    return traced_out_rho.reshape([rho_dim, rho_dim]) 

"""Define the U3 gate with three parameters."""
def U3Gate(theta, phi, lam):
    aa=torch.tensor([
            [torch.cos(theta/2), -torch.exp(1j*lam)*torch.sin(theta/2)],
            [torch.exp(1j*phi)*torch.sin(theta/2), torch.exp(1j*(lam+phi))*torch.cos(theta/2)]
        ]) 
    return aa

def IGate():
    aa=torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0]
        ]) 
    return aa
   
def XGate():
    aa=torch.tensor([
            [0.0, 1.0],
            [1.0, 0.0]
        ])  
    return aa

def YGate():
    aa=torch.tensor([
            [0.0, -1j],
            [1j, 0.0]
        ])  
    return aa

def ZGate():
    aa=torch.tensor([
            [1.0, 0.0],
            [0.0, -1.0]
        ])  
    return aa

def generate_cangate(params:Any,
                     num_qubits:int,
                     start_index: int,
                     end_index:int) -> Any:

    # for can gate
    all_gate=[XGate(),YGate(),ZGate()]
    iparam=0
    circ_all=torch.zeros(2**num_qubits,2**num_qubits)
    for igate in all_gate:
        circ=torch.cos(params[iparam]/2)
        circ2=-1j*torch.sin(params[iparam]/2)

        for i in range(num_qubits):
            circ=torch.kron(circ,IGate())

            if i == start_index or i == end_index:
                circ2=torch.kron(circ2,igate)
            else:
                circ2=torch.kron(circ2,IGate())

        if iparam==0:
            circ_all=circ_all+circ+circ2    
        else:
            circ_all=torch.matmul((circ+circ2),circ_all)

        iparam += 1

    return circ_all     
     
def generate_canonical_circuit_all_neurons(qnn_arch_all:list,
                                           params: List[Any],
                                           layer: int) -> Any:  
    """
    Creates a QuantumCircuit containing the parameterized CAN gates (plus single qubit U3 gates).
    The definition of the CAN gates is taken from https://arxiv.org/abs/1905.13311.
    The parameters should have length self.qnn_arch[0]*self.qnn_arch[1]*6 + self.qnn_arch[1]*6.

    Args:
        params (List[Any]): List of parameters for the parametrized gates. ParameterVector or List[float].
        layer (int): Index of the current output layer.
    Returns:
        QuantumCircuit: Quantum circuit containing all the parameterized gates of the respective layer.
    """
    # sub-architecture of the layer (length 2)
    qnn_arch = qnn_arch_all[layer-1:layer+1]
    # number of qubits required for the layer
    num_qubits = qnn_arch[0]+qnn_arch[1]
   
    # add U3s to input qubits
    circ = add_one_qubit_gates(qnn_arch[0], qnn_arch[1], params[:qnn_arch[0]*3])
  
    # loop over all neurons
    for i in range(qnn_arch[1]):
        # parameters of the respective "neuron gates"
        # (can be larer than needed, overflow will be ignored)
        neuron_params = params[qnn_arch[0]*3 + qnn_arch[0]*3*i:]
        # iterate over all input neurons and apply CAN gates
        for j in range(qnn_arch[0]):
            tx, ty, tz = neuron_params[j*3:(j+1)*3]
            circ=torch.matmul(generate_cangate([2*tx,2*ty,2*tz], num_qubits, j, qnn_arch[0]+i),circ)          

    return circ

def add_one_qubit_gates(num_qubits: int,
                        sec_qubits: int, 
                        params: List[float]) -> Any:
    """
    Adds U3 gates to each qubit in the quantum register.

    Args:
        q_reg (QuantumRegister): The quantum register containing the qubits.
        params (List[flaot]): List of parameters (used for the one qubit gates). Should be a multiple of 3.

    Returns:
        QuantumCircuit: The given quantum circuit including the application of one qubit gates.
    """
    circ=U3Gate(params[0], params[1], params[2])
    for i in range(1,num_qubits):
        circ=torch.kron(circ,U3Gate(params[i*3], params[i*3+1], params[i*3+2]))

    for i in range(sec_qubits):
        circ=torch.kron(circ,IGate())
    return circ

def save_data(all_params_epochs: Optional[Any] = None, 
              plot_list_cost: Optional[List[List[Union[Union[int,float],float]]]] = None) -> None:
    """
    Saves and plots the given data to a file.

    Args:
        all_params_epochs (Optional[List[List[Union[float, List[float]]]], optional): The ansatz's parameters per epoch. Defaults to None.
        plot_list_cost (float, optional): Training cost per epoch. Defaults to None.
    """    

    if type(all_params_epochs) is np.ndarray: 
        np.savetxt("./params.txt",all_params_epochs)
    
    if plot_list_cost: 
        with open('./energy.txt','a') as f:
            np.savetxt(f,plot_list_cost)

#        plot_cost(plot_list_cost)

def plot_cost(plot_list: List[List[float]],
              filename: str = "energy.pdf") -> None:
    """
    Plot the cost versus the learning epoch.

    Args:
        plot_list (List[List[float]]]): List of [epoch, cost].
        filename (str, optional): The name of the output file. Defaults to "cost.pdf".
    """     
    if not (isinstance(plot_list, np.ndarray)): plot_list = np.array(plot_list)
    try:
        plt.figure()
        plt.plot(plot_list[:,0], "--o", color="b", label="QML", ms=4)
        plt.plot(plot_list[:,1], "--o", color="r", label="FCI", ms=4)
        plt.xlabel("Epoch")
        plt.ylabel("Energy (Ha)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("./{}".format(filename))
    except:
        logger.warning("Cost could not be plotted. An error occured.")
    finally:
        plt.close()
