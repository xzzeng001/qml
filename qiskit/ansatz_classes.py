# own modules
import save_data as sd # type: ignore
import training
import sys
import utils

# --- QISKIT ---
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer, execute, assemble# type: ignore
from qiskit.quantum_info.operators import Operator # type: ignore
from qiskit.circuit import ParameterVector # type: ignore
from qiskit.providers.ibmq.managed import IBMQJobManager # type: ignore
import qiskit.providers.aer.noise as noise # type: ignore

from qiskit.quantum_info import Pauli
from qiskit.opflow.expectations import AerPauliExpectation,PauliExpectation
from qiskit.opflow.state_fns import StateFn,CircuitStateFn
from qiskit.opflow.list_ops import SummedOp
from qiskit.opflow.primitive_ops import PauliOp
from qiskit import Aer
from qiskit.aqua.operators.legacy import TPBGroupedWeightedPauliOperator 

# additional math libs
import numpy as np # type: ignore
from scipy.constants import pi # type: ignore

from typing import Union, Optional, List, Tuple, Any, Dict
import itertools

import logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class Ansatz_Pool:
    """
    Architecture #9: https://www.notion.so/9-CAN-gates-applied-directly-to-the-corresponding-qubits-without-swaps-e1a16986754d4dfebc3bad33bac939a5
    """
    def __init__(self,
                name: str = 'dpnn',
                n_qubits: Optional[int] = None,
                n_orb: Optional[int] = None,
                n_orb_occ: Optional[int] = None,
                occ_indices_spin: Optional[List[int]] = None, 
                qnn_arch: Optional[List[int]]=None,
                measurement_method: Optional[str] = 'swap_trick',
                hamiltonian: Optional[Any] = None,
                params: Optional[str] = None,
                num_entangle: Optional[int]=1,
                excited_ranks: Optional[str]='sd',
                nterms: Optional[int] = 1,
                ep_cont: Optional[List[float]] = None,
                fci_e: Optional[float]=0,
                gate_error_probabilities: Optional[dict] = None):
        """
        Initialized the network class.

        Args:
            qnn_arch (List[int]): The QNN architecture (e.g. [2,2]).
            measurement_method (str, optional): The measurement method. Defaults to 'swap_trick'.
            params (Optional[str], optional): The initial network parameters. Defaults to None (will be randomly generated).
        """
        self.name = name
        self.auxillary_qubits = 1 if measurement_method == "swap_trick" else 0 # 1 for swap_trick 
        if self.name == 'dqnn':       
            assert qnn_arch[0]==qnn_arch[-1], "Not a valid QNN-Architecture."
            self.qnn_arch = qnn_arch
            self.num_qubits = qnn_arch[0]
            self.num_params = sum([qnn_arch[l]*qnn_arch[l+1]*3 + (qnn_arch[l])*3 for l in range(len(qnn_arch)-1)]) + qnn_arch[-1]*3 # total number of parameters
            self.params_per_layer = [self.qnn_arch[l]*self.qnn_arch[l+1]*3 + (self.qnn_arch[l])*3 for l in range(len(qnn_arch)-1)] # number of parameters per layer
            self.params = generate_network_parameters(num_params=self.num_params, load_from=params)

            # calculate the required qubits
            a=np.array(self.qnn_arch)
            ind = np.argpartition(a, -2)[-2:]
            self.required_qubits = a[ind[0]]+a[ind[1]]+self.auxillary_qubits # required number of qubits
#            self.required_qubits = sum(self.qnn_arch)+self.auxillary_qubits # required number of qubits
        elif self.name == 'ucc':
            self.excited_ranks=excited_ranks
            self.num_qubits = n_qubits
            self.n_orb = n_orb   # the number of orbital available
            self.n_orb_occ = n_orb_occ # the number of occupied orbitals
            self.required_qubits = n_qubits + self.auxillary_qubits # required number of qubits
            self.params = []
        elif self.name == 'hardware':
            self.num_qubits = n_qubits
            self.required_qubits = n_qubits + self.auxillary_qubits # required number of qubits 
            self.num_entangle=num_entangle
            self.num_params = (num_entangle+1)*self.num_qubits*3
            self.params = generate_network_parameters(num_params=self.num_params, load_from=params)     

        self.occ_indices_spin = occ_indices_spin
        self.meas_method = measurement_method
        self.tpb_grouping=False if measurement_method == "swap_trick" else True  # whether use the tpb grouping method
        self.num_groups = 1 # the number of grouped hamiltonian 
        self.group_member=[] # the list of members for each group
        self.nterms = nterms   # the number of hamiltonian 
        self.hamiltonian=hamiltonian  # the required hamiltonian for the measurement
        self.fci_e = fci_e   # the target energy
        self.min_e=99999     # the minimum energy for all the step 
        # circuit types
        self.psi: Any =None   # the ansatz of circuit
        self.ep_circuits: List[Any] = [] # expectation value circuits
        self.ep_cont = ep_cont  # the constant of hamiltonian
        self.ep_ncircuits = nterms - 1    # number of circuits for expectation      
        self.param_vector: List[Any] = [] # parameter vector (network parameters)
        # gate nosie study
        self.gate_error_probabilities=gate_error_probabilities
        self.noise_model: Any = None
        self.coupling_map: Any = None
        
    def __str__(self):
        """
        The network's description.

        Returns:
            string: Network description.
        """ 
        if self.name == 'dqnn':       
            return "Ansatz_DQNN of the form: {}".format(self.qnn_arch)
        elif self.name == 'ucc' or self.name == 'hardware': 
            return "Ansatz_name: {}".format(self.name+self.excited_ranks)
        elif self.name == 'hardware':
            return "Ansatz_name: {}".format(self.name)
        
    def update_params(self, params: Optional[Union[str, List[float]]] = None):
        """
        Update the network parameters.

        Args:
            params (Optional[Union[str, List[float]]], optional): The new network parameters. Defaults to None.
        """       
        if params is not None:
            # if params are not given -> initialize them randomly
            self.params = generate_network_parameters(num_params=self.num_params, load_from=params) if (isinstance(params,str)) else params 
        else:
            logger.warning('Params could not be updated because given params are None.')
          
    def circuit(self,
                params: Union[List[float], Any],
                draw_circ: bool = False,
                ep_gates: Optional[str]=None) -> QuantumCircuit:
        """
        Creates the quantum circuit.

        Args:
            state_pair (List[np.ndarray]): Training pairs (used for initialization).
            params (Union[List[float], Any]): Network parameters.
            draw_circ (bool, optional): Whether the circuit should be drawn. Defaults to False.

        Returns:
            QuantumCircuit: The quantum circuit.
        """    
        # initialize the quantum circuit
        circ, q_reg, c_reg = init_quantum_circuit(self.required_qubits, 1 if self.meas_method == "swap_trick" else 0)

        input_index=[i for i in range(self.auxillary_qubits,self.auxillary_qubits + self.qnn_arch[0])]
        tmp_out=[i for i in range(self.required_qubits)]
        for ii in input_index:
           tmp_out.remove(ii)

        # going through each output layer
        for layer in range(len(self.qnn_arch)-1):
            output_index=tmp_out[0:self.qnn_arch[layer+1]]
            # the resepctive parameters
            layer_params = params[np.sign(layer)*sum(self.params_per_layer[:layer]):sum(self.params_per_layer[:layer+1])]

            index_all=input_index+output_index
            # the respective qubit register
            in_and_output_register = q_reg[index_all]

            # append subcircuit connecting all neurons of (layer+1) to layer
            circ.append(self.generate_canonical_circuit_all_neurons(layer_params, layer=layer+1, draw_circ=draw_circ).to_instruction(), in_and_output_register)

            circ.reset(input_index)
            input_index=output_index
            tmp_out=[i for i in range(self.required_qubits)]
            for ii in input_index:
               tmp_out.remove(ii)

        # add last U3s to all output qubits (last layer)
        circ = add_one_qubit_gates(circ, q_reg[-self.qnn_arch[-1]:], params[-self.qnn_arch[-1]*3:])

        if self.ep_ncircuits > 100:
            file_name='circuit_all.txt'
        else:
            file_name='circuit_all.png'

        if (draw_circ):
            # draw the sub-circuit
            sd.draw_circuit(circ.decompose(), filename=file_name)

#        print('parameters:',circ.parameters)
        # add expectation value measurement
        if self.meas_method == "swap_trick":
            circ = add_ep_measurement(circ, q_reg, c_reg, self.required_qubits-self.num_qubits, ep_gates=ep_gates)
        return circ

    def ucc_circuit(self,
                params: Union[List[float], Any],
                ucc_operator_pool: list=None,
                draw_circ: bool = False,
                ep_gates: Optional[str]=None) -> QuantumCircuit:
        """
        Creates the ucc quantum circuit.

        Args:
            state_pair (List[np.ndarray]): Training pairs (used for initialization).
            params (Union[List[float], Any]): Network parameters.
            draw_circ (bool, optional): Whether the circuit should be drawn. Defaults to False.

        Returns:
            QuantumCircuit: The quantum circuit.
        """    
        # initialize the quantum circuit
        circ, q_reg, c_reg = init_quantum_circuit(self.required_qubits, 1 if self.meas_method == "swap_trick" else 0)

        # genrate the HF circuit for the ucc ansatz
        circ=generate_hartree_fock_circuit(circ, q_reg, self.occ_indices_spin)

        # Generate the equivalent circuit for the time evolution operator 
        circ=generate_time_evolution_circuit(circ=circ, q_reg=q_reg, c_reg=c_reg, qubit_operators_list=ucc_operator_pool, amplitudes=params)

        # add expectation value measurement
        if self.meas_method == "swap_trick":
            circ = add_ep_measurement(circ, q_reg, c_reg, self.required_qubits-self.num_qubits, ep_gates=ep_gates)
        return circ

    def hardware_circuit(self,
                params: Union[List[float], Any],
                draw_circ: bool = False,
                ep_gates: Optional[str]=None) -> QuantumCircuit:
        """
        Creates the hardware-efficient quantum circuit.

        Args:
            state_pair (List[np.ndarray]): Training pairs (used for initialization).
            params (Union[List[float], Any]): Network parameters.
            draw_circ (bool, optional): Whether the circuit should be drawn. Defaults to False.

        Returns:
            QuantumCircuit: The quantum circuit.
        """    
        # initialize the quantum circuit
        circ, q_reg, c_reg = init_quantum_circuit(self.required_qubits, 1 if self.meas_method == "swap_trick" else 0)

        # add last U3s to all output qubits (last layer)
        circ = add_one_qubit_gates(circ, q_reg, params[:self.num_qubits*3])

        # loop over all neurons
        for i in range(self.num_entangle):
            # parameters of the respective "neuron gates"
            neuron_params = params[self.num_qubits*3*(i+1):self.num_qubits*3*(i+2)]
            # genrate the entangled circuits for the hardware-efficient ansatz
            circ=generate_entangle_circuit(circ, neuron_params, q_reg, self.num_qubits)

        # add expectation value measurement
        if self.meas_method == "swap_trick":
            circ = add_ep_measurement(circ, q_reg, c_reg, self.required_qubits-self.num_qubits, ep_gates=ep_gates)
        return circ

    def transpile_circuits(self,
                           circuits: List[QuantumCircuit],
                           backend: Any,
                           optimization_level: int = 3,
                           idx_circuit: Optional[int]=1,
                           draw_circ: bool = False,
                           Ham_op: Optional[Any]=None,
                           save_info: Optional[bool]=True) -> List[QuantumCircuit]:
        """
        Transpiles the given circuits.

        Args:
            circuits (List[QuantumCircuit]): The QuantumCircuits which should be transpiled.
            backend (Any): The Backend for which the transpilation should be optimized.
            optimization_level (int, optional): The optimization level of the transpilation. Defaults to 3.
            draw_circ (bool, optional): Whether the transpiled circuit should be drawn. Defaults to False.

        Returns:
            List[QuantumCircuit]: The transpiled quantum circuits.
        """
        # set the backend, coupling map, basis gates and optimization level of gate_error_probabilities (if given)
        transpile_backend = backend if not self.gate_error_probabilities else None
        transpile_coupling_map = None if not self.gate_error_probabilities else self.coupling_map
        transpile_basis_gates = None if not self.gate_error_probabilities else self.noise_model.basis_gates
        # optimization level should be ever 0,1,2 or 3
        if not optimization_level in [0,1,2,3]: 
            logger.warning("Optimization level out of bounds. An optimization level of 3 will be used.")
            optimization_level = 3
        transpile_optlvl = 1 if backend.name() == 'qasm_simulator' or backend.name() == 'aer_simulator' and not self.gate_error_probabilities else optimization_level

        # transpile the quantum circuits
        transpiled_circuits = transpile(circuits, backend=transpile_backend, 
            optimization_level=transpile_optlvl, coupling_map=transpile_coupling_map, 
            basis_gates=transpile_basis_gates, seed_transpiler=0)
        
        # function should return a list of quantum circuits
        if not isinstance(transpiled_circuits, list): transpiled_circuits = [transpiled_circuits]

        if draw_circ:

            if self.ep_ncircuits > 100:
                file_suffix='.txt'
            else:
                file_suffix='.png'

            # draw a single circuit
#            file_idx='circuit_'+str(idx_circuit)+file_suffix
#            sd.draw_circuit(circuits[0], filename=file_idx)
            # draw a single transpiled circuits
            file_idx='transpiled_circuit_'+str(idx_circuit)+file_suffix
            sd.draw_circuit(transpiled_circuits[0], filename=file_idx)

        # save the depth and number of operations of the transpiled circuit
        if save_info:
            sd.save_execution_info(transpilation_info="depth: {}, count_ops: {}".format(transpiled_circuits[0].depth(), transpiled_circuits[0].count_ops()),num_params=self.num_params)

#        transpiled_circuits[0].qasm('circuit.txt')
#        sys.exit(0)

        if Ham_op == None:
            return transpiled_circuits
        else:
#            Ham_op=SummedOp([PauliOp(Pauli('IIIIIIII'),-0.2879450776014974)])
#            print('Ham_op:',Ham_op)
            self.psi=transpiled_circuits[0]
            psi = CircuitStateFn(transpiled_circuits[0])
            measurable_expression = StateFn(Ham_op, is_measurement=True).compose(psi)
#            if self.num_qubits < 10:
#                expectation = AerPauliExpectation().convert(measurable_expression)
#            else:
#                expectation = PauliExpectation().convert(measurable_expression)
            expectation = PauliExpectation(group_paulis=True).convert(measurable_expression)
             
#            print('circuit',expectation.to_circuit)

            return expectation
 
    def execute_circuits(self,
                         circuits: List[QuantumCircuit], 
                         backend: Any, 
                         shots: int = 2**14, 
                         **unused_args: Any) -> Any:
        """
        Executes the QuantumCircuits using the Backend.

        Args:
            circuits (List[QuantumCircuit]): The QuantumCircuits which should be executed.
            backend (Any): The backend for the execution.
            shots (int, optional): The number of shots used for the execution. Defaults to 2**14.

        Returns:
            Any: The measurement results (list of counts).
        """        
        if not self.gate_error_probabilities:
            if "qasm_simulator" in backend.name() or "aer_simulator" in backend.name():
                # simulator or simulated backend
                qobj_circuits = assemble(circuits, backend=backend, shots=shots)
                job = backend.run(qobj_circuits)
                return job.result().get_counts()
            # real device execution
            # use the IBMQJobManager to break the circuits into multiple jobs
            job_manager = IBMQJobManager()
            job = job_manager.run(circuits, backend=backend, shots=shots)
            result = job.results()
            return [result.get_counts(i) for i in range(len(circuits))]
        # gate_error_probability is given -> use the specific function
        return execute_noise_simulation(self, circuits, self.gate_error_probabilities, shots)

    def generate_canonical_circuit_all_neurons(self,
                                               params: List[Any],
                                               layer: int,
                                               draw_circ: bool = False) -> QuantumCircuit:  
        """
        Creates a QuantumCircuit containing the parameterized CAN gates (plus single qubit U3 gates).
        The definition of the CAN gates is taken from https://arxiv.org/abs/1905.13311.
        The parameters should have length self.qnn_arch[0]*self.qnn_arch[1]*6 + self.qnn_arch[1]*6.

        Args:
            params (List[Any]): List of parameters for the parametrized gates. ParameterVector or List[float].
            layer (int): Index of the current output layer.
            draw_circ (bool, optional): Whether the sub-circuit should be drawn. Defaults to False.

        Returns:
            QuantumCircuit: Quantum circuit containing all the parameterized gates of the respective layer.
        """
        # sub-architecture of the layer (length 2)
        qnn_arch = self.qnn_arch[layer-1:layer+1]
        # number of qubits required for the layer
        num_qubits = qnn_arch[0]+qnn_arch[1]
        
        # initialize the quantum circuit
        circ, q_reg, _ = init_quantum_circuit(num_qubits, name="Layer {}".format(layer))
        
        # add U3s to input qubits
        circ = add_one_qubit_gates(circ, q_reg[:qnn_arch[0]], params)
        
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

##        if self.ep_ncircuits > 20:
##            file_name='circuit-layer_{}.txt'
##        else:
##            file_name='circuit-layer_{}.png'
##
##        if (draw_circ):
##            # draw the sub-circuit
##        #   print('come to here new!')
##            sd.draw_circuit(circ, filename=file_name.format(layer))
        return circ

    def reshape_params_for_plotting(self,
                                    param_per_epoch: List[List[float]]) -> List[List[dict]]:
        """
        Reshapes the list of parameter-lists. Includes information about the plotting title and values for parameter plotting.
        The return value is used for the specific plotting routine.

        Args:
            param_per_epoch (Any): List of parameters per training epoch.

        Returns:
            List[List[Dict]]: [[{title: "Layer 1, Preparation", params: [params per epoch]}, ... (neuronwise)], ... (layerwise)]
        """
        all_layers: List[List[dict]] = []
        # go through each output layer
        for layer in range(len(self.qnn_arch)-1):
            all_neurons: List[dict] = []
            # parameters of this layer (per training epoch)
            layer_params = param_per_epoch[np.sign(layer)*sum(self.params_per_layer[:layer]):sum(self.params_per_layer[:layer+1])]
            for j in range(self.qnn_arch[layer+1]+1):
                neuron_dict: dict = {}
                if j == 0:
                    # first layer (includes U3 gates)
                    neuron_dict['title'] = 'Layer {}, Preparation'.format(layer+1)
                    neuron_dict['params'] = layer_params[:self.qnn_arch[layer]*3]
                else:
                    neuron_dict["title"] = 'Layer {}, Neuron {}'.format(layer+1, j)
                    neuron_dict['params'] = layer_params[self.qnn_arch[layer]*3:][3*self.qnn_arch[layer]*(j-1):3*self.qnn_arch[layer]*j]
                    if layer == len(self.qnn_arch)-2:
                        # last layer (includes last U3 gates)
                        if j-1 < self.qnn_arch[-1]-1:
                            neuron_dict['params'] += param_per_epoch[-(self.qnn_arch[-1]-(j-1))*3:-(self.qnn_arch[-1]-(j-1)-1)*3]
                        else:
                            neuron_dict['params'] += param_per_epoch[-3:]
                all_neurons.append(neuron_dict)
            all_layers.append(all_neurons)      
        return all_layers          

    def tpb_grouping_hamiltonian(self) -> Any:
        """
        Grouping the hamiltonian based on the tensor product basis
        """
        Ham_op=[]
        Ham_op2=[]

        ii = 0
        for term in self.hamiltonian.terms:
            op_str=''
            # for constant terms
            if len(term) == 0:
                for ik in range(self.required_qubits):
                    op_str+='I'
            # for full terms with num_qubits
            elif len(term) == self.num_qubits:
                for ep_gate in term:
                    if ep_gate[1] == 'X':
                        op_str=op_str+'X'
                    elif ep_gate[1] == 'Y':
                        op_str=op_str+'Y'
                    elif ep_gate[1] == 'Z':
                        op_str=op_str+'Z'
            # for the case which Pauli terms less than num_qubits
            # add the identity gate between them
            elif len(term) < self.num_qubits:
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
                if idx_prev - 1 <= self.num_qubits - 1:
                    idiff = self.num_qubits - idx_prev
                    for jj in range(idiff):
                        op_str=op_str+'I'
                # for the error case
                else: 
                   print('Error in number of gates in tpb_grouping!!!') 
                   sys.exit(0)

            if len(term) != 0 and self.required_qubits != self.num_qubits:
                for ik in range(self.required_qubits-self.num_qubits):
                    op_str+='I'

            Ham_op.append(PauliOp(Pauli(op_str),self.ep_cont[ii]))
            Ham_op2.append([self.ep_cont[ii],Pauli(op_str)])
            ii += 1

        rrt=TPBGroupedWeightedPauliOperator(paulis=Ham_op2,basis=None)
        rr2=rrt.simplify()
        str_details=rrt.sorted_grouping(rr2).print_details()
        list_details=str_details.split('\n')
        self.num_groups=rrt.sorted_grouping(rr2).num_groups
        ik=0
        for ii in range(self.num_groups):
            # for the weight
            nn=int(list_details[ii+ik].split(' ')[2].split('(')[1].split(')')[0])
            self.group_member.append(nn)
            ik=ik+nn

        Ham_op=SummedOp(Ham_op)

        return Ham_op

def generate_network_parameters(param_range: Optional[Union[float, List[float]]] = 2*pi,
                                num_params: Optional[int] = None,
                                load_from: Optional[str] = None) -> List[float]:
    """
    Generate random network parameters (in specific range and with specific length). They are used for the parameterized gates.

    Args:
        param_range (Optional[Union[float, List[float]]], optional): Range of the random parameters. Defaults to 2*pi.
        num_params (Optional[int], optional): Number of parameters. Defaults to None.
        load_from (Optional[str], optional): Name of a file containing the parameters. Defaults to None.

    Returns:
        List[float]: The generated network parameters.
    """    
    if load_from:
        # filename is given -> load parameter from file
        all_params = np.loadtxt(load_from)
        if len(np.shape(all_params)) < 2: all_params = np.array([all_params])
        # length of loaded parameters, should match num_params
        if len(all_params[0][0:]) != num_params:
            logger.error("Loaded parameters have different size ({}) than expected ({})."
                .format(len(all_params[0][1:]), num_params))
            raise
        # if more than one paremeter list is given: display warning
#        if (len(all_params) > 0): logger.warning('Your loaded parameters {} have more than one parameter set available. choose the last one.'.format(load_from))
        return all_params[len(all_params)-1][0:]
    if isinstance(param_range, list):
        # param_range is a list -> consists of lower and upper bound
        return np.random.uniform(low=param_range[0], high=param_range[1], size=(num_params)) # Range of parameters, e.g. [-pi,pi] or [-1,1]
    # param_range is a float -> range = [0, param_range]
    return np.random.uniform(high=param_range, size=(num_params))


def construct_noise_model(ansatz: Union[Ansatz_Pool],
                          device_name: str) -> None:
    """
    Constructs the noise model for the gate_error_probabilities of the respectice network.

    Args:
        ansatz (Union[Ansatz_Pool]): The ansatz's class. 
    """    
#    provider = training.get_provider()
#    backend = provider.get_backend('qasm_simulator')
    backend = Aer.get_backend(device_name)
    ansatz.coupling_map = backend.configuration().coupling_map
        
    noise_model = noise.NoiseModel(["u3", "rxx", "ryy", "rzz"])
    for gate, value in ansatz.gate_error_probabilities.items():
        error = noise.depolarizing_error(*value)
        noise_model.add_all_qubit_quantum_error(error, gate)
    ansatz.noise_model = noise_model


def construct_and_transpile_circuits(ansatz: Union[Ansatz_Pool],
                                     optimization_level: int = 3,
                                     ucc_operator_pool: Optional[list]=None,
                                     device_name: Optional[str] = None,
                                     simulator: Optional[bool] = None,
                                     draw_circ: Optional[bool] = None,
                                     save_info: Optional[bool] = True) -> None:
    """
    Constructs and transpiles the quantum circuits for all given training and validation pairs.

    Args:
        ansatz (Union[Ansatz_Pool]): The ansatz's class.
        training_pairs (List[List[np.ndarray]]): The list of training pairs.
        optimization_level (int, optional): The optimization level used for the transpilation. Defaults to 3.
        device_name (Optional[str], optional): The name of the backend. Defaults to None.
        simulator (Optional[bool], optional): Whether a simulator (or simulated device) should be used. Defaults to None.
    """    
    if simulator:
        # simulator or simulated device
        device = training.DEVICE or training.get_device(ansatz, bool(simulator), device_name=device_name, do_calibration=False)
        training.DEVICE = device
    else:
        # real device
        device = training.get_device(ansatz, bool(simulator), device_name=device_name, do_calibration=False)
    
    # parameter initialization
    ansatz.param_vector = ParameterVector("p", ansatz.num_params)

    # change the draw method
    sd.determine_draw_method(nH=ansatz.ep_ncircuits)

    if ansatz.name == 'dqnn':
        # grouping the Hamiltonian based on Tensor Product Basis(TPB)
        if ansatz.tpb_grouping:
            Ham_op=ansatz.tpb_grouping_hamiltonian() 
            tp_circs= [ansatz.circuit(params=ansatz.param_vector, draw_circ=draw_circ)]
            expectation=ansatz.transpile_circuits(tp_circs, backend=device, Ham_op=Ham_op, optimization_level=optimization_level,draw_circ=draw_circ,save_info=save_info)
            ansatz.ep_ncircuits=len(expectation)
            return expectation
        else:
            ii=0
            for term in ansatz.hamiltonian.terms:
                if len(term) == 0:
                    continue
                else:
                    ep_gates=''
                    for ind_gate in term:
                       ep_gates += str(ind_gate[1])+str(ind_gate[0])+' '

                ep_gates=ep_gates.strip() 
                tp_circs = [ansatz.circuit(params=ansatz.param_vector, ep_gates=ep_gates, draw_circ=draw_circ)]
                ansatz.ep_circuits.append(ansatz.transpile_circuits(tp_circs, backend=device, optimization_level=optimization_level, idx_circuit=ii, draw_circ=draw_circ,save_info=save_info))
                ii+=1
            return None

    elif ansatz.name == 'ucc':
        # grouping the Hamiltonian based on Tensor Product Basis(TPB)
        if ansatz.tpb_grouping:
            Ham_op=ansatz.tpb_grouping_hamiltonian()
            tp_circs= [ansatz.ucc_circuit(params=ansatz.param_vector, ucc_operator_pool=ucc_operator_pool, draw_circ=draw_circ)]
            expectation=ansatz.transpile_circuits(tp_circs, backend=device, Ham_op=Ham_op, optimization_level=optimization_level,draw_circ=draw_circ,save_info=save_info)
            ansatz.ep_ncircuits=len(expectation)
            return expectation
        else:
            ii=0
            for term in ansatz.hamiltonian.terms:
                if len(term) == 0:
                    continue
                else:
                    ep_gates=''
                    for ind_gate in term:
                       ep_gates += str(ind_gate[1])+str(ind_gate[0])+' '

                ep_gates=ep_gates.strip() 
                tp_circs = [ansatz.ucc_circuit(params=ansatz.param_vector, ucc_operator_pool=ucc_operator_pool, ep_gates=ep_gates, draw_circ=draw_circ)]
                ansatz.ep_circuits.append(ansatz.transpile_circuits(tp_circs, backend=device, optimization_level=optimization_level, idx_circuit=ii, draw_circ=draw_circ,save_info=save_info))
                ii+=1
            return None
     
    elif ansatz.name == 'hardware':
       # grouping the Hamiltonian based on Tensor Product Basis(TPB)
       if ansatz.tpb_grouping:
           Ham_op=ansatz.tpb_grouping_hamiltonian() 
           tp_circs= [ansatz.hardware_circuit(params=ansatz.param_vector, draw_circ=draw_circ)]
           expectation=ansatz.transpile_circuits(tp_circs, backend=device, Ham_op=Ham_op, optimization_level=optimization_level,draw_circ=draw_circ)
           ansatz.ep_ncircuits=len(expectation)
           return expectation
       else:
           ii=0
           for term in ansatz.hamiltonian.terms:
               if len(term) == 0:
                   continue
               else:
                   ep_gates=''
                   for ind_gate in term:
                      ep_gates += str(ind_gate[1])+str(ind_gate[0])+' '

               ep_gates=ep_gates.strip() 
               tp_circs = [ansatz.harware_circuit(params=ansatz.param_vector, ep_gates=ep_gates, draw_circ=draw_circ)]
               ansatz.ep_circuits.append(ansatz.transpile_circuits(tp_circs, backend=device, optimization_level=optimization_level, idx_circuit=ii, draw_circ=draw_circ))
               ii+=1
           return None

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

def add_input_state_initialization(circ: QuantumCircuit,
                                   q_reg: QuantumRegister,
                                   num_qubits_per_state: int,
                                   state_pair: List[np.ndarray]) -> QuantumCircuit:
    """
    Adds the state initialization to the given QuantumCircuit.

    Args:
        circ (QuantumCircuit): The quantum circuit.
        q_reg (QuantumRegister): The quantum register (qubits for the initialization).
        num_qubits_per_state (int): The number of qubits for each state.
        state_pair (List[np.ndarray]): A pair of states.

    Returns:
        QuantumCircuit: The given quantum circuit including state initializations.
    """        
    # initialize qubits
    circ.initialize(state_pair[0],[q_reg[i] for i in range(num_qubits_per_state)])
    return circ

def add_one_qubit_gates(circ: QuantumCircuit, 
                        q_reg: QuantumRegister, 
                        params: List[float],
                        u3: bool = True) -> QuantumCircuit:
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
        if u3:
            circ.u(params[i*3], params[i*3+1], params[i*3+2], qubit)
        else:
            circ.rx(params[i*3], qubit)
            circ.ry(params[i*3+1], qubit)
            circ.rx(params[i*3+2], qubit)
    return circ

def add_ep_measurement(circ: QuantumCircuit,
                             q_register: QuantumRegister,
                             c_register: ClassicalRegister,
                             idx_qubits: int,
                             ep_gates: str) -> QuantumCircuit:
    """
    Adds a expectation measurement to the given QuantumCircuit.
    The classical 'SWAP trick'
    Args:
        circ (QuantumCircuit): The quantum circuit.
        q_register (QuantumRegister): The quantum register containing the qubits.
        c_register (ClassicalRegister): The classical register (the measurement result is stored here).
        idx_qubits (int): The index of output qubits
        ep_gates (str): The description of the Pauli gate information
    Returns:
        QuantumCircuit: The given quantum circuit including the expectation measurement.
    """
    circ.h(q_register[0])

    # add the constrolled Pauli gates to the target qubits
    if len(ep_gates) > 2:
        nn=len(ep_gates.split(" "))
        for ii in range(nn):
            iterm=ep_gates.split(" ")[ii]
            idx=int(iterm[1:])
            if iterm[0] == 'X':
                circ.cx(q_register[0], q_register[idx+1])
            elif iterm[0] == 'Y':
                circ.cy(q_register[0], q_register[idx+1])          
            elif iterm[0] == 'Z':
                circ.cz(q_register[0], q_register[idx+1])
    else:
        iterm=ep_gates
        idx=int(iterm[1:])
        if iterm[0] == 'X':
            circ.cx(q_register[0], q_register[idx+1])
        elif iterm[0] == 'Y':
            circ.cy(q_register[0], q_register[idx+1])
        elif iterm[0] == 'Z':
            circ.cz(q_register[0], q_register[idx+1])
       
    circ.h(q_register[0])
    # measurement of the ancillary qubit
    circ.measure(q_register[0], c_register[0])

    return circ   

def generate_entangle_circuit(circ: QuantumCircuit,
                              params: List[float],
                              q_register: QuantumRegister,
                              num_qubits: int)-> QuantumCircuit:
    """
    Generate the entangle circuits

    Returns:
        A qiskit quantum circuit object.

    Notes:
        The circuit will use n_qubits Qubits,
    """
    for i in range(num_qubits):
        circ.u3(params[i*3], params[i*3+1], params[i*3+2], q_register[i])
    for i in range(num_qubits - 1):
        circ.cx(i, i + 1)

    return circ

def generate_hartree_fock_circuit(circ: QuantumCircuit,
                                  q_register: QuantumRegister,
                                  spin_orbital_occupied_indices: list)-> QuantumCircuit:
    """
    Generate the state preparation circuit, that is the Hartree Fock state.

    Returns:
        A qiskit quantum circuit object.

    Notes:
        The circuit is constructed under the assumption that the
        Fermion-to-Qubit mapping is the Jordan-Wigner transformation.

    Notes:
        The circuit will use n_qubits Qubits,
    """

    for i in spin_orbital_occupied_indices:
        circ.x(q_register[i])

    return circ

def generate_time_evolution_circuit(circ: QuantumCircuit,
                                    q_reg: QuantumRegister,
                                    c_reg: ClassicalRegister,
                                    qubit_operators_list: list,
                                    amplitudes: np.ndarray)-> QuantumCircuit:
    """
    Generate the equivalent circuit for the time evolution operator:
        op = sum(qubit_operators_list)
        time_evolution = exp(op)
    where op is an openfermion.QubitOperator and should be an anti-Hermitian
    operator. Since the operators are decomposed under Pauli basis,
    the condition "anti-Hermitian" also indicates that the coefficients should
    be pure imaginary numbers.

    Returns:
        A qiskit quantum circuit object.

    Notes:
        The circuit corresponds to the first-order Trotter decomposition:
            exp(A + B) approximately equals to exp(A)exp(B)
        therefore, the time evolution exp(op) can be approximated by:
            exp(op0) exp(op1) exp(op2) ...
    """
    HY_matrix = 2 ** 0.5 / 2. * np.array(
        [[1.0, -1.j],
         [1.j, -1.0]]
    )
    n_operators = len(qubit_operators_list)
    single_trottered_gate_list = []
    for i in range(n_operators):
        single_trottered_gate_list += \
            utils.decompose_trottered_qubitOp(
                qubit_operators_list[i], i)

    n_gates = len(single_trottered_gate_list)
    for gate_idx in range(n_gates):
        gate_info = single_trottered_gate_list[gate_idx]
        gate_symbol = gate_info[0]
        if gate_symbol == "H":
            qubit_idx = gate_info[1]
            circ.h(q_reg[qubit_idx])
        elif gate_symbol == "HY":
            qubit_idx = gate_info[1]
            circ.u3(np.pi / 2., np.pi / 2., np.pi / 2.,q_reg[qubit_idx])
        elif gate_symbol == "CNOT":
            ctrl_idx = gate_info[1][0]
            target_idx = gate_info[1][1]
            circ.cx(q_reg[ctrl_idx], q_reg[target_idx])
        elif gate_symbol == "RZ":
            qubit_idx = gate_info[1][0]
            pre_fact = gate_info[1][1]
            amplitude_idx = gate_info[1][2]
            amplitude_rz = amplitudes[amplitude_idx] * pre_fact
            amplitude_rz = amplitude_rz
            circ.rz(amplitude_rz,q_reg[qubit_idx])

    return circ
