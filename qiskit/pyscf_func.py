import functools

import numpy as np
import scipy
import scipy.sparse
import pyscf
import pyscf.lo
import pyscf.cc
import pyscf.fci
import openfermion
import sys

from openfermion.utils import count_qubits

from typing import Union, Optional, List, Tuple, Any, Dict


def init_scf(geometry: list,
                with_fci: bool=True, 
                basis: str = "sto-3g", 
                 spin: int = 0,
               charge: int=0):
    '''
    calculate the one-body and two-body integral based on the Pyscf
    obtain the Hamiltoian
    '''
    molecule = pyscf.gto.M(
            atom = geometry,
            basis = basis,
            spin = spin,
            charge = charge,
            symmetry = True
    )

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

    energy=energy_RCCSD
    if with_fci:
        mf_fci = pyscf.fci.FCI(mf)
        energy_FCI = mf_fci.kernel()[0]
        print("FCI energy: %20.16f Ha" % (energy_FCI))
        energy=energy_FCI

    n_orb = molecule.nao_nr()
    n_orb_occ = sum(molecule.nelec) // 2
    occ_indices_spin = [i for i in range(n_orb_occ * 2)]
    hcore = mf.get_hcore()
    mo_coeff = mf.mo_coeff
    one_body_mo = functools.reduce(np.dot, (mo_coeff.T, hcore, mo_coeff))
    two_body_mo = pyscf.ao2mo.restore(1, pyscf.ao2mo.get_mo_eri(
        molecule, mo_coeff, compact=False), 
        n_orb
    )
    one_body_int = np.zeros([n_orb * 2] * 2)
    two_body_int = np.zeros([n_orb * 2] * 4)

    for p in range(n_orb):
        for q in range(n_orb):
            one_body_int[2 * p][2 * q] = one_body_mo[p][q]
            one_body_int[2 * p + 1][2 * q + 1] = one_body_mo[p][q]
            for r in range(n_orb):
                for s in range(n_orb):
                    two_body_int[2 * p][2 * q][2 * r][2 * s] = two_body_mo[p][s][q][r]
                    two_body_int[2 * p + 1][2 * q + 1][2 * r + 1][2 * s + 1] = two_body_mo[p][s][q][r]
                    two_body_int[2 * p + 1][2 * q][2 * r][2 * s + 1] = two_body_mo[p][s][q][r]
                    two_body_int[2 * p][2 * q + 1][2 * r + 1][2 * s] = two_body_mo[p][s][q][r]
    
    hamiltonian_fermOp_1 = openfermion.FermionOperator()
    hamiltonian_fermOp_2 = openfermion.FermionOperator()

    for p in range(n_orb * 2):
        for q in range(n_orb * 2):
            hamiltonian_fermOp_1 += openfermion.FermionOperator(
                ((p, 1), (q, 0)),
                one_body_int[p][q]
            )
    for p in range(n_orb * 2):
        for q in range(n_orb * 2):
            for r in range(n_orb * 2):
                for s in range(n_orb * 2):
                    hamiltonian_fermOp_2 += openfermion.FermionOperator(
                        ((p, 1), (q, 1), (r, 0), (s, 0)),
                        two_body_int[p][q][r][s] * 0.5
                    )

    hamiltonian_fermOp_1 = openfermion.normal_ordered(hamiltonian_fermOp_1)
    hamiltonian_fermOp_2 = openfermion.normal_ordered(hamiltonian_fermOp_2)
    hamiltonian_fermOp = hamiltonian_fermOp_1 + hamiltonian_fermOp_2
    hamiltonian_fermOp += energy_nuc

    hamiltonian_qubitOp = openfermion.jordan_wigner(hamiltonian_fermOp)
    n_qubits = openfermion.count_qubits(hamiltonian_qubitOp)

    # TODO: Change the first return value to openfermion's MolecularData
    return molecule, n_qubits, n_orb, n_orb_occ, occ_indices_spin, \
           hamiltonian_fermOp, hamiltonian_qubitOp, energy

def obtain_Hamiltonian(geometry: list,
                          basis: str = "sto-3g", 
                          spin: int = 0,
                          charge: int=0, 
                          dist: float=1.5,
                          with_fci: bool=True,
                          BK_reduce: bool=True):
    '''
    obtain the Hamiltoian based on the openfermion
    '''
    molecule, n_qubits, n_orb, n_orb_occ, occ_indices_spin, \
    hamiltonian_fermOp, hamiltonian_qubitOp,e\
        = init_scf(geometry, with_fci, basis, spin,charge)

    if BK_reduce:
        hamiltonian_qubitOp_reduced = openfermion.symmetry_conserving_bravyi_kitaev(hamiltonian_fermOp, int(n_orb) * 2, sum(molecule.nelec))
        hamiltonian_qubitOp_reduced,nterms,ep_cont = chop_to_real(hamiltonian_qubitOp_reduced)
    else:
        hamiltonian_qubitOp_reduced,nterms,ep_cont = chop_to_real(hamiltonian_qubitOp)

    n_qubits=count_qubits(hamiltonian_qubitOp_reduced)

    print("n_qubits, n_orb, n_orb_occ, occ_indices_spin\n",n_qubits, n_orb, n_orb_occ, occ_indices_spin)
    return e, n_qubits, n_orb, n_orb_occ, occ_indices_spin, nterms, ep_cont, hamiltonian_qubitOp_reduced

def chop_to_real(hamiltonian_qubitOp: Any, adapt: Optional[bool]=False):
    '''
    chop the imaginary part of the weighted terms in hamiltonian
    '''
    nterms=len(hamiltonian_qubitOp.terms)
    new_terms={}

    new_cont=[]
    for term in hamiltonian_qubitOp.terms:
        ep_cont=hamiltonian_qubitOp.terms[term]
        if adapt:
            new_terms[term]=ep_cont
            new_cont.append(ep_cont)
        else:
            if isinstance(ep_cont,np.complex): 
               new_terms[term]=ep_cont.real
               new_cont.append(ep_cont.real)
            else:
               new_terms[term]=ep_cont
               new_cont.append(ep_cont)

    hamiltonian_qubitOp.terms=new_terms

    return hamiltonian_qubitOp, nterms, new_cont
