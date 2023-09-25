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
from openfermion.utils import count_qubits

#dist=4.0
#geometry = [["O", [0.00000000,0.40111200,0.00000000]],
#            ["H", [0.97092700,-3.4758230,0.00000000]],
#            ["H", [-0.9709270,0.26692700,0.00000000]]]
#
#basis  = "sto3g"
#spin   = 0
#charge = 0
#
#E_fci,qubits,n_orb,n_orb_occ,occ_indices_spin,ham, = obtain_Hamiltonian(geometry,basis,spin,charge,with_fci=True, BK_reduce=True) 

#if energy < E_fci:
#   E_fci=energy
if os.path.exists('ham.npy'):
    ham=np.load('ham.npy',allow_pickle=True).item()
    H=convert_observable(ham)
else:
    H=convert_observable(ham)
    np.save('ham.npy',ham)

qubits = count_qubits(ham)
#H=convert_observable(ham) 
# define the network structure
network=[qubits,2]

n_qubits_tot=sum(network)
ncycle=1

n_tot_params=3*network[0]
for icycle in range(ncycle):
    n_tot_params += network[0]*network[1]

#n_tot_params +=3*network[0]

print('The Architecture of network: ',network)
print('The cyle of each layer: ',ncycle)
print('The total parameters of network:',n_tot_params)

sys.stdout.flush()
#print('test:',sum(network[:2]))

def circuit(params):
   qml.BasisState(np.zeros(n_qubits_tot),wires=range(n_qubits_tot))

   nparams=0
   # for the input gates
   for i in range(network[0]):
      qml.U3(params[nparams+i*3], params[nparams+i*3+1], params[nparams+i*3+2], wires=i)

   nparams=3*network[0]

   for icycle in range(ncycle):

      # for the intermediate layer
      for i in range(network[0]):
         # parameters of the respective "neuron gates"
         # (can be larer than needed, overflow will be ignored)
         neuron_params = params[nparams:]
         # iterate over all input neurons and apply CAN gates
         for j in range(network[1]):
#            tx, ty, tz = neuron_params[j*3:(j+1)*3]
            tx = neuron_params[j]  
            qml.IsingXX(2*tx, wires=(sum(network[:1])+j,i))
#            qml.IsingYY(2*ty, wires=(sum(network[:1])+j,i))
#            qml.IsingZZ(2*tz, wires=(sum(network[:1])+j,i))
#            print('two target:',sum(network[:n_layer-ilayer-2])+j,sum(network[:n_layer-ilayer-1])+i)
         nparams += 1 #3*network[1]

#      # add single qubit gates
#      for i in range(network[1]):
#         qml.U3(params[nparams+i*3], params[nparams+i*3+1],
#                params[nparams+i*3+2], wires=qubits+i)
#
#      nparams += 3*network[1]

#      # for the last excahnge part
#      for i in range(network[1]):
#         # parameters of the respective "neuron gates"
#         # (can be larer than needed, overflow will be ignored)
#         neuron_params = params[nparams:]
#         # iterate over all input neurons and apply CAN gates
#         for j in range(network[0]):
#            tx, ty, tz = neuron_params[j*3:(j+1)*3]
#            qml.IsingXX(2*tx, wires=(j,sum(network[:1])+i))
#            qml.IsingYY(2*ty, wires=(j,sum(network[:1])+i))
#            qml.IsingZZ(2*tz, wires=(j,sum(network[:1])+i))
##            print('two target-2:',j,sum(network[:n_layer-1])+i)
#         nparams += 3*network[0]
#

   # add single qubit gates
#   for i in range(network[0]):
#      qml.U3(params[nparams+i*3], params[nparams+i*3+1],
#             params[nparams+i*3+2], wires=i)

#assert torch.cuda.is_available()
cpu_device = torch.device("cpu") 

dev = qml.device("default.qubit.torch", wires=n_qubits_tot)

@qml.qnode(dev,interface="torch", diff_method="backprop")
def cost_fn(params):
   circuit(params)
   return qml.expval(H)

#opt=qml.AdamOptimizer(stepsize=0.01)

if os.path.exists('params.txt'):
    theta=torch.tensor(np.loadtxt('params.txt'),requires_grad=True,dtype=torch.float64,device=cpu_device)
else:
    theta=torch.tensor(np.random.uniform(high=2*np.pi,size=(n_tot_params)),requires_grad=True,dtype=torch.float64,device=cpu_device)

#start_time=time()

#end_time=time()

#print('total_time:',end_time-start_time)

#sys.exit(0)

max_iterations=10000
optimizer = torch.optim.Adam([theta],lr=0.01)
scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

#optimizer = torch.optim.NAdam([theta])
#optimizer = torch.optim.SGD([theta],lr=0.1, momentum=0.9)
#optimizer = torch.optim.Adadelta([theta],lr=0.1)

prev_energy=cost_fn(theta).detach().numpy()
#e_fci=torch.tensor(E_fci,requires_grad=False,dtype=torch.float64,device=cpu_device)
for n in range(max_iterations):
#   theta,prev_energy=opt.step_and_cost(cost_fn,theta)
   r_energy=cost_fn(theta)
   energy=r_energy
   optimizer.zero_grad()
   energy.backward()
   optimizer.step()

   scheduler.step()

   r_energy=r_energy.detach().numpy()
#   conv=r_energy-E_fci

   with open('energy.txt','a') as f:
       np.savetxt(f,[[n,r_energy]])

   if r_energy < prev_energy:
#       torch.save(theta,'params.pt')
       np.savetxt("params.txt",[theta.detach().numpy()])
       prev_energy=r_energy

#   print(f"Step = {n}, Energy = {energy:.8f} Ha, error = {conv:.8f} Ha")

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
