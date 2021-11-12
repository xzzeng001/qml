from pennylane import qchem
import numpy as np
import pennylane as qml
import os,sys
import pyscf
import pyscf.cc
import pyscf.fci


dist=0.7
symbols=["H","H","H","H"] 
coordinates=np.array([0.,0.,0,0.,0.,dist,0.,0.,2*dist,0.,0.,3*dist])

ang_to_au=1.0/0.529177249

coordinates=coordinates*ang_to_au

geometry = [["H", [0.0, 0.0, 0.0]],
            ["H", [0.0, 0.0, 1.0*dist]],
            ["H", [0.0, 0.0, 2.0*dist]],
            ["H", [0.0, 0.0, 3.0*dist]]]

basis  = "sto3g"
spin   = 0
charge = 0

# run the pyscf to obtain the FCI energy
molecule = pyscf.gto.M(
         atom = geometry,
         basis = basis,
         spin = spin,
         charge = charge,
         symmetry = True)

mf = pyscf.scf.RHF(molecule)
print("Running RHF...")
mf.kernel()
print("Running RCCSD")
mf_cc = pyscf.cc.RCCSD(mf)
mf_cc.kernel()

energy_RHF = mf.e_tot
energy_RCCSD = mf_cc.e_tot
energy_nuc = molecule.energy_nuc()
print("Hartree-Fock energy: %20.16f Ha" % (energy_RHF))
print("CCSD energy: %20.16f Ha" % (energy_RCCSD))
mf_fci = pyscf.fci.FCI(mf)
energy_FCI = mf_fci.kernel()[0]
print("FCI energy: %20.16f Ha" % (energy_FCI))
E_fci=energy_FCI


H, qubits = qchem.molecular_hamiltonian(
            symbols,
            coordinates,
            charge=0,
            mult=1,
            basis='sto-3g',
            active_electrons=4,
            active_orbitals=4)

print('Number of qubits=',qubits)
#print('The Hamiltonian is ',H)

dev = qml.device("default.qubit", wires=2*qubits)

#electrons=2
#hf=qml.qchem.hf_state(electrons,qubits)
#print(hf)

def circuit(params,qubits):
   qml.BasisState(np.zeros(2*qubits),wires=range(2*qubits))
   
   # add single qubit gates
   for i in range(qubits):
      qml.U3(params[i*3], params[i*3+1], params[i*3+2], wires=qubits+i)

   for i in range(qubits):
      # parameters of the respective "neuron gates"
      # (can be larer than needed, overflow will be ignored)
      neuron_params = params[qubits*3 + qubits*3*i:]
      # iterate over all input neurons and apply CAN gates
      for j in range(qubits):
          tx, ty, tz = neuron_params[j*3:(j+1)*3]
          qml.IsingXX(2*tx, wires=(j,qubits+i))
          qml.IsingYY(2*ty, wires=(j,qubits+i))
          qml.IsingZZ(2*tz, wires=(j,qubits+i))

   # add single qubit gates
   for i in range(qubits):
      qml.U3(params[3*(qubits+qubits*qubits)+i*3], params[3*(qubits+qubits*qubits)+i*3+1], 
             params[3*(qubits+qubits*qubits)+i*3+2], wires=i)

@qml.qnode(dev)
def cost_fn(params):
   circuit(params,qubits)
   return qml.expval(H)

opt=qml.AdagradOptimizer(stepsize=0.01)
if os.path.exists('params.txt'):
   theta=np.array(np.loadtxt('params.txt'))
else:
   theta=np.random.rand(3*qubits*qubits+6*qubits)

energy=[cost_fn(theta)]

max_iterations=1000

for n in range(max_iterations):
   theta,prev_energy=opt.step_and_cost(cost_fn,theta)
   energy=cost_fn(theta)
   conv=energy-E_fci

   with open('energy.txt','a') as f:
      np.savetxt(f,[[dist,energy,E_fci,conv]])

   if energy < prev_energy:
      np.savetxt("params.txt",[theta])

   print(f"Step = {n}, Energy = {energy:.8f} Ha, error = {conv:.8f} Ha")

#print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
#print("\n" f"Optimal value of the circuit parameter = {angle[-1]:.4f}")

#import matplotlib.pyplot as plt
#
#fig = plt.figure()
#fig.set_figheight(5)
#fig.set_figwidth(12)
#
## Full configuration interaction (FCI) energy computed classically
##E_fci = -1.136189454088
#
#print('Final accuracy: ',energy[-1]-E_fci)
#
## Add energy plot on column 1
#ax1 = fig.add_subplot(121)
#ax1.plot(range(n + 2), energy, "go-", ls="dashed")
#ax1.plot(range(n + 2), np.full(n + 2, E_fci), color="red")
#ax1.set_xlabel("Optimization step", fontsize=13)
#ax1.set_ylabel("Energy (Hartree)", fontsize=13)
#ax1.text(0.5, -1.1176, r"$E_\mathrm{HF}$", fontsize=15)
#ax1.text(0, -1.1357, r"$E_\mathrm{FCI}$", fontsize=15)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#
#'''
## Add angle plot on column 2
#ax2 = fig.add_subplot(122)
#ax2.plot(range(n + 2), angle, "go-", ls="dashed")
#ax2.set_xlabel("Optimization step", fontsize=13)
#ax2.set_ylabel("Gate parameter $\\theta$ (rad)", fontsize=13)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#
#plt.subplots_adjust(wspace=0.3, bottom=0.2)
#'''
#plt.show()  
