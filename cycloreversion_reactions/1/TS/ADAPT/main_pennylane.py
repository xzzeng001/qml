from pennylane import qchem
import numpy as np
import pennylane as qml
import sys,os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import pyscf
import pyscf.cc
import pyscf.fci
from time import time
from pyscf_func import obtain_Hamiltonian
from pennylane_qchem.qchem import convert_observable 


#if energy < E_fci:
#   E_fci=energy
if os.path.exists('ham.npy'):
    ham=np.load('ham.npy',allow_pickle=True).item()
    H=convert_observable(ham)
else:
    H=convert_observable(ham)
    np.save('ham.npy',ham)

print('number of elec: ',n_elec)
print("number of qubits: ",qubits)

hf_state = qml.qchem.hf_state(n_elec, qubits)

# Generate single and double excitations
singles, doubles = qml.qchem.excitations(n_elec, qubits)

# Map excitations to the wires the UCCSD circuit will act on
s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

# Define the device
dev = qml.device("default.qubit", wires=qubits)

print("Number of singles: ", len(singles))
print("Number of doubles: ", len(doubles))

singles_excitations = [qml.SingleExcitation(0.0, x) for x in singles]
doubles_excitations = [qml.DoubleExcitation(0.0, x) for x in doubles]
operator_pool = doubles_excitations + singles_excitations

dev = qml.device("default.qubit", wires=qubits)
@qml.qnode(dev)
def circuit():
    [qml.PauliX(i) for i in np.nonzero(hf_state)[0]]
    return qml.expval(H)

opt = qml.optimize.AdaptiveOptimizer()
for i in range(len(operator_pool)):
    circuit, energy, gradient = opt.step_and_cost(circuit, operator_pool, drain_pool=True)
    if gradient < 1e-2:
        break   

def circuit_1(params, excitations):
    qml.BasisState(hf_state, wires=range(qubits))

    for i, excitation in enumerate(excitations):
        if len(excitation) == 4:
            qml.DoubleExcitation(params[i], wires=excitation)
        else:
            qml.SingleExcitation(params[i], wires=excitation)
    return qml.expval(H)

tol=1e-2
dev = qml.device("default.qubit", wires=qubits)
cost_fn = qml.QNode(circuit_1, dev, interface="autograd")

circuit_gradient = qml.grad(cost_fn, argnum=0)

params = [0.0] * len(doubles)
grads = circuit_gradient(params, excitations=doubles)

doubles_select = [doubles[i] for i in range(len(doubles)) if abs(grads[i]) > tol]

opt = qml.GradientDescentOptimizer(stepsize=0.5)

params_doubles = np.zeros(len(doubles_select))

for n in range(20):
    params_doubles = opt.step(cost_fn, params_doubles, excitations=doubles_select)

def circuit_2(params, excitations, gates_select, params_select):
    qml.BasisState(hf_state, wires=range(qubits))

    for i, gate in enumerate(gates_select):
        if len(gate) == 4:
            qml.DoubleExcitation(params_select[i], wires=gate)
        elif len(gate) == 2:
            qml.SingleExcitation(params_select[i], wires=gate)

    for i, gate in enumerate(excitations):
        if len(gate) == 4:
            qml.DoubleExcitation(params[i], wires=gate)
        elif len(gate) == 2:
            qml.SingleExcitation(params[i], wires=gate)
    return qml.expval(H)

cost_fn2 = qml.QNode(circuit_2, dev, interface="autograd")
circuit_gradient = qml.grad(cost_fn2, argnum=0)
params = [0.0] * len(singles)

grads = circuit_gradient(
    params,
    excitations=singles,
    gates_select=doubles_select,
    params_select=params_doubles
)

print('grads: ',grads)
singles_select = [singles[i] for i in range(len(singles)) if abs(grads[i]) > tol]

cost_fn3 = qml.QNode(circuit_1, dev, interface="autograd")

params = np.zeros(len(doubles_select + singles_select))

gates_select = doubles_select + singles_select

print("Number of select singles: ", len(singles_select))
print("Number of select doubles: ", len(doubles_select))

for n in range(20):
    params, energy = opt.step_and_cost(cost_fn3, params, excitations=gates_select)
    print('n and error: ',n,energy-E_fci)
    if abs(energy-E_fci) < 1.6e-3:
        break

@qml.qnode(dev)
def circuit_specs(params, excitations):
   qml.BasisState(hf_state, wires=range(qubits))

   for i, excitation in enumerate(excitations):
       if len(excitation) == 4:
           qml.DoubleExcitation.compute_decomposition(params[i], wires=excitation)
       else:
           qml.SingleExcitation.compute_decomposition(params[i], wires=excitation)
   return qml.expval(H)

print('spec1: ',qml.specs(circuit_specs)(params, gates_select))


