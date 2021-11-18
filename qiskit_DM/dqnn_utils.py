# own modules
import sys
import matplotlib.pyplot as plt

# --- QISKIT ---
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info.operators import Operator # type: ignore

from qiskit.quantum_info import Pauli
from qiskit.opflow.list_ops import SummedOp
from qiskit.opflow.primitive_ops import PauliOp

# additional math libs
import numpy as np # type: ignore
from scipy.constants import pi # type: ignore

from typing import Union, Optional, List, Tuple, Any, Dict
import itertools

import logging
logger = logging.getLogger(__name__)

def init_quantum_circuit(num_qubits, num_cbits=0, name: Optional[str] = None) -> Tuple[QuantumCircuit, QuantumRegister, ClassicalRegister]:
    """
    Initializes a QuantumCircuit using num_qubits qubits and num_cbits classical bits.

    Args:
        num_qubits ([type]): The number of qubits.
        num_cbits (int, optional): The number of classical bits. Defaults to 0.
        name (Optional[str], optional): The quantum circuit's name. Defaults to None.

    Returns:
        Tuple[QuantumCircuit, QuantumRegister, ClassicalRegister]: The initialized QuantumCircuit and its QuantumRegister and ClassicalRegister
    """    
    # init register for quantum and classical bits
    q_register = QuantumRegister(num_qubits, 'q')
    c_register = ClassicalRegister(num_cbits, 'c') if num_cbits > 0 else None

    # init quantum circuit
    circ = QuantumCircuit(num_qubits, num_cbits, name=name) if c_register else QuantumCircuit(num_qubits, name=name)
    return circ, q_register, c_register

def generate_canonical_circuit_all_neurons(qnn_arch_all:list,
                                           params: List[Any],
                                           layer: int) -> QuantumCircuit:  
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
   
    # initialize the quantum circuit
    circ, q_reg, _ = init_quantum_circuit(num_qubits)
    
    # add U3s to input qubits
    circ = add_one_qubit_gates(circ, q_reg[:qnn_arch[0]], params[:qnn_arch[0]*3])
   
    # loop over all neurons
    for i in range(qnn_arch[1]):
        # parameters of the respective "neuron gates"
        # (can be larer than needed, overflow will be ignored)
        neuron_params = params[qnn_arch[0]*3 + qnn_arch[0]*3*i:]
        # iterate over all input neurons and apply CAN gates
        for j in range(qnn_arch[0]):
            tx, ty, tz = neuron_params[j*3:(j+1)*3]
            circ.rxx(2*tx, q_reg[j], q_reg[qnn_arch[0]+i])
            circ.ryy(2*ty, q_reg[j], q_reg[qnn_arch[0]+i])
            circ.rzz(2*tz, q_reg[j], q_reg[qnn_arch[0]+i])

    return circ

def add_one_qubit_gates(circ: QuantumCircuit, 
                        q_reg: QuantumRegister, 
                        params: List[float]) -> QuantumCircuit:
    """
    Adds U3 gates (if u3=True else RX, RY, RX gates) to each qubit in the quantum register.

    Args:
        circ (QuantumCircuit): The quantum circuit.
        q_reg (QuantumRegister): The quantum register containing the qubits.
        params (List[flaot]): List of parameters (used for the one qubit gates). Should be a multiple of 3.
        u3 (bool): Whether U3 gates should be used. Defaults to True.

    Returns:
        QuantumCircuit: The given quantum circuit including the application of one qubit gates.
    """
    for i, qubit in enumerate(q_reg):
        circ.u(params[i*3], params[i*3+1], params[i*3+2], qubit)
    return circ

def transform_the_hamiltonian(hamiltonian:Any,
                              num_qubits:int,
                              ep_cont:Any) -> Any:
    """
    transform the hamiltonian to the qiskit operator base
    """
    Ham_op=[]

    ii = 0
    for term in hamiltonian.terms:
        op_str=''
        # for constant terms
        if len(term) == 0:
            for ik in range(num_qubits):
                op_str+='I'
        # for full terms with num_qubits
        elif len(term) == num_qubits:
            for ep_gate in term:
                if ep_gate[1] == 'X':
                    op_str=op_str+'X'
                elif ep_gate[1] == 'Y':
                    op_str=op_str+'Y'
                elif ep_gate[1] == 'Z':
                    op_str=op_str+'Z'
        # for the case which Pauli terms less than num_qubits
        # add the identity gate between them
        elif len(term) < num_qubits:
            idx_prev=0
            for ep_gate in term:
                idx=int(ep_gate[0])
                if idx > idx_prev:
                    idiff = idx - idx_prev
                    for kk in range(idiff):
                        op_str=op_str+'I'
                # choose the right gate
                if ep_gate[1] == 'X':
                    op_str=op_str+'X'
                elif ep_gate[1] == 'Y':
                    op_str=op_str+'Y'
                elif ep_gate[1] == 'Z':
                    op_str=op_str+'Z'
                idx_prev = idx + 1

            # for the last check !!
            if idx_prev - 1 <= num_qubits - 1:
                idiff = num_qubits - idx_prev
                for jj in range(idiff):
                    op_str=op_str+'I'
            # for the error case
            else: 
               print('Error in number of gates !!!') 
               sys.exit(0)

        Ham_op.append(PauliOp(Pauli(op_str),ep_cont[ii]))
        ii += 1
    return SummedOp(Ham_op)

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
        np.savetxt("./energy.txt",plot_list_cost)
        plot_cost(plot_list_cost)

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
