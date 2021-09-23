import save_data as sd

import scipy as sc # type: ignore
import numpy as np # type: ignore
from typing import Union, Optional, List, Tuple, Any
import sys

def randomQubitUnitary(numQubits):
    """
    returns a unitary  2^(numQubits)×2^(numQubits)-matrix
    as a numpy array (np.ndarray) that is the tensor product
    of numQubits factors. 
    Before orthogonalization, it's elements are randomly picked
    out of a normal distribution.
    """
    dim = 2**numQubits
    #Make unitary matrix
    res = sc.random.normal(size=(dim,dim)) + 1j * sc.random.normal(size=(dim,dim))
    res = sc.linalg.orth(res)
    #Return
    return res

def randomQubitState(numQubits):
    dim = 2**numQubits
    #Make normalized state
    res = sc.random.normal(size=(dim,1)) + 1j * sc.random.normal(size=(dim,1))
    res = (1/sc.linalg.norm(res)) * res
    #Return
    return res.flatten()

def obtainQubitState(numQubits):
    qubit1 = np.array([[0],[1]]).T
    qubit0 = np.array([[1],[0]]).T
    res=qubit1
    for idx in range(1,numQubits):
        res = np.kron(qubit1, res)    
    return res

def generate_training_data(num_qubits: int) -> Tuple[List[List[np.ndarray]]]:    
    """
    generate_training_data is used to prepare a given number of training pairs
    for a given network architecture such that one training pair can be used
    to directly initialize the first 2m qubits of the circuit (m=network.num_qubits)

    Args:
    num_qubits: int

    Returns:
    training pairs: list[[QubitState, Unitary*QubitState]]
    """
    num_training_pairs = 1 
    
    # load training states
    input_states: List[np.ndarray] = []
    input_states=obtainQubitState(num_qubits)

    assert all((len(input_state) == 2**num_qubits) for input_state in input_states), 'Dimension of input training states does not match network architecture'

    return np.array([input_states])
