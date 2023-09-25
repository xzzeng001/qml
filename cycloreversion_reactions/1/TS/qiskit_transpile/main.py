#from qiskit import transpile, Aer
from typing import Union, Optional, List, Tuple, Any, Dict
import numpy as np
from circuit import UCCSD_circuit, HAA_circuit, KMA_circuit
from qiskit.providers.aer import QasmSimulator

# for UCCSD ansatz circuit
#-------------------------
##num_params, circ=UCCSD_circuit(10,6)

# for HAA ansatz
#-------------------------
network=[10,2]
ncycle=2
num_params, circ=HAA_circuit(network,ncycle)

# for KMA ansatz
#-------------------------
##qubits=10
##ncycle=4
##num_params, circ=HAA_circuit(qubits,ncycle) 

##print('depth of circuit 1 and number of Cz :', circ.depth(), circ.num_nonlocal_gates())
##
#print(circ.qasm())
#circ.draw(output='mpl',filename='1.eps')
##
### set the backend, coupling map, basis gates and optimization level of gate_error_probabilities (if given)
##transpile_backend = Aer.get_backend('qasm_simulator')
##
##transpile_coupling_map = None 
##transpile_basis_gates = ['h','cz','p']
### optimization level should be ever 0,1,2 or 3
##optimization_level = 2
##
### transpile the quantum circuits
##transpiled_circuits = transpile(circ, backend=transpile_backend,
##    optimization_level=optimization_level, coupling_map=transpile_coupling_map,
##    basis_gates=transpile_basis_gates, seed_transpiler=0)
##
##print('number of params: ',num_params)
##print('depth of circuit 2  and number of Cz :', transpiled_circuits.depth(), transpiled_circuits.num_nonlocal_gates())
##
###transpiled_circuits.draw(output='mpl',filename='2.png')
