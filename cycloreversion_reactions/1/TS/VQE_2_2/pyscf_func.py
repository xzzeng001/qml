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


def init_scf_old(geometry: list,
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
#            symmetry = True
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

def init_scf(geometry, basis="sto-3g", spin=0,
             freeze_indices_spatial=[],
             active_indices_spatial=[],
             run_fci: bool = True,
             localized_orbitals: str = None,
             use_symmetry: bool = False,
             override_symmetry_group: str = None,
             fermion_to_qubit_mapping: str = "jw"):
    """
    Generate the system Hamiltonian and other quantities for a give molecule.

    Args:
        geometry (list): The structure of the molecule.
        basis (str): Basis set for SCF calculations.
        spin (int): Describes multiplicity of the molecular system.
        freeze_indices_spatial (list): Occupied indices (frozen orbitals)
            of spatial orbitals.
        active_indices_spatial (list): Active indices of spatial
            orbitals.
        run_fci (bool): Whether FCI calculation is performed.
        localized_orbitals (str): Whether to use localized orbitals. If
            is None, no localization if performed.
        use_symmetry (bool): Whether to use symmetry and return the character
            table of orbitals. Exclusive with localized_orbitals.
        override_symmetry_group (str): Override the symmetry point group
            determined by PySCF.
        fermion_to_qubit_mapping (str): The fermion-to-qubit mapping
            for Hamiltonian.

    Returns:
        molecule (pyscf.gto.M object): Contains various properties
            of the system.
        n_qubits (int): Number of qubits in the Hamiltonian.
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        occ_indices_spin (int): Occupied indices of spin orbitals.
        hamiltonian_fermOp (openfermion.FermionOperator): Fermionic
            Hamiltonian.
        hamiltonian_qubitOp (openfermion.QubitOperator): Qubit Hamiltonian
            under JW transformation.
        orbsym (numpy.ndarray): The irreducible representation of each
            spatial orbital. Only returns when use_symmetry is True.
        prod_table (numpy.ndarray): The direct production table of orbsym.
            Only returns when use_symmetry is True.

    """

    if localized_orbitals is not None:
        if use_symmetry is True:
            raise ValueError("Symmetry cannot be used together \
with orbital localization!")

    molecule = pyscf.gto.M(
        atom=geometry,
        basis=basis,
        spin=spin
    )

    if use_symmetry:
        if override_symmetry_group is not None:
            molecule = pyscf.gto.M(
                atom=geometry,
                basis=basis,
                spin=spin,
                symmetry=override_symmetry_group
            )
        else:
            molecule = pyscf.gto.M(
                atom=geometry,
                basis=basis,
                spin=spin,
                symmetry=True
            )
        print("Use symmetry. Molecule point group: %s" % (molecule.topgroup))

    mf = pyscf.scf.RHF(molecule)
    print("Running RHF...")
    mf.kernel()
    mo_coeff = mf.mo_coeff
    if localized_orbitals is not None:
        if localized_orbitals in ["iao", "IAO"]:
            print("Use IAO localization.")
            mo_occ = mf.mo_coeff[:, mf.mo_occ > 0]
            a = pyscf.lo.iao.iao(molecule, mo_occ)
            a = pyscf.lo.vec_lowdin(a, mf.get_ovlp())
            mo_occ = a.T.dot(mf.get_ovlp().dot(mo_occ))
            mo_coeff = a.copy()
        elif localized_orbitals in ["ibo", "IBO"]:
            print("Use IBO localization.")
            mo_occ = mf.mo_coeff[:, :]
            a = pyscf.lo.iao.iao(molecule, mo_occ)
            a = pyscf.lo.vec_lowdin(a, mf.get_ovlp())
            a = pyscf.lo.ibo.ibo(molecule, mo_occ, iaos=a, s=mf.get_ovlp())
            a = pyscf.lo.vec_lowdin(a, mf.get_ovlp())
            mo_occ = a.T.dot(mf.get_ovlp().dot(mo_occ))
            mo_coeff = a.copy()
        else:
            raise ValueError("Localization orbital %s \
not supported!" % (localized_orbitals))

    print("Running RCCSD")
    mf_cc = pyscf.cc.RCCSD(mf)
    mf_cc.kernel()

    energy_RHF = mf.e_tot
    energy_RCCSD = mf_cc.e_tot
    energy_nuc = molecule.energy_nuc()
    print("Hartree-Fock energy: %20.16f Ha" % (energy_RHF))
    print("CCSD energy: %20.16f Ha" % (energy_RCCSD))

    if run_fci:
        mf_fci = pyscf.fci.FCI(mf)
        energy_fci = mf_fci.kernel()[0]
        print("FCI energy: %20.16f Ha" % (energy_fci))

    n_orb = molecule.nao_nr()
    n_orb_occ = sum(molecule.nelec) // 2
    occ_indices_spin = [i for i in range(n_orb_occ * 2)]
    hcore = mf.get_hcore()
    one_body_mo = functools.reduce(np.dot, (mo_coeff.T, hcore, mo_coeff))
    two_body_mo = pyscf.ao2mo.restore(1, pyscf.ao2mo.get_mo_eri(
        molecule, mo_coeff, compact=False),
        n_orb
    )
    core_correction = 0.0

    if (len(freeze_indices_spatial) == 0) \
            and (len(active_indices_spatial) == 0):
        pass
    elif (len(active_indices_spatial) != 0):
        n_orb = len(active_indices_spatial)
        n_orb_occ = (sum(molecule.nelec) -
                     2 * len(freeze_indices_spatial)) // 2
        occ_indices_spin = [i for i in range(n_orb_occ * 2)]
        one_body_mo_new = np.copy(one_body_mo)
        for p in freeze_indices_spatial:
            core_correction += 2. * one_body_mo[p][p]
            for q in freeze_indices_spatial:
                core_correction += (2. * two_body_mo[p][q][q][p] -
                                    two_body_mo[p][q][p][q])
        for uu in active_indices_spatial:
            for vv in active_indices_spatial:
                for ii in freeze_indices_spatial:
                    one_body_mo_new[uu][vv] += (
                        2. * two_body_mo[ii][ii][uu][vv] -
                        two_body_mo[ii][vv][uu][ii]
                    )
        one_body_mo = one_body_mo_new[np.ix_(
            active_indices_spatial, active_indices_spatial)]
        two_body_mo = two_body_mo.transpose(0, 2, 3, 1)[np.ix_(
            active_indices_spatial, active_indices_spatial,
            active_indices_spatial, active_indices_spatial)]
        two_body_mo = two_body_mo.transpose(0, 3, 1, 2)
    else:
        print("active_indices_spatial must not be empty \
if freeze_indices_spatial is non-empty !")
        raise ValueError

    one_body_int = np.zeros([n_orb * 2] * 2)
    two_body_int = np.zeros([n_orb * 2] * 4)

    for p in range(n_orb):
        for q in range(n_orb):
            one_body_int[2 * p][2 * q] = one_body_mo[p][q]
            one_body_int[2 * p + 1][2 * q + 1] = one_body_mo[p][q]
            for r in range(n_orb):
                for s in range(n_orb):
                    two_body_int[2 * p][2 * q][2 * r][2 * s] = \
                        two_body_mo[p][s][q][r]
                    two_body_int[2 * p + 1][2 * q + 1][2 * r + 1][2 * s + 1] = \
                        two_body_mo[p][s][q][r]
                    two_body_int[2 * p + 1][2 * q][2 * r][2 * s + 1] = \
                        two_body_mo[p][s][q][r]
                    two_body_int[2 * p][2 * q + 1][2 * r + 1][2 * s] = \
                        two_body_mo[p][s][q][r]

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
    hamiltonian_fermOp += energy_nuc + core_correction

    hamiltonian_qubitOp = None
    if fermion_to_qubit_mapping is not None:
        if fermion_to_qubit_mapping == "jw":
            hamiltonian_qubitOp = openfermion.jordan_wigner(hamiltonian_fermOp)
        else:
            raise NotImplementedError("Fermion-to-qubit mapping {} not \
implemented.".format(fermion_to_qubit_mapping))
    n_qubits = openfermion.count_qubits(hamiltonian_fermOp)

    if use_symmetry:
        orbsym = mf.orbsym
        prod_table = pyscf.symm.direct_prod(
            np.arange(len(pyscf.symm.symm_ops(molecule.topgroup))),
            np.arange(len(pyscf.symm.symm_ops(molecule.topgroup))),
            molecule.topgroup)
        return molecule, n_qubits, n_orb, n_orb_occ, occ_indices_spin, \
            hamiltonian_fermOp, hamiltonian_qubitOp, \
            orbsym, prod_table

    # TODO: Change the first return value to openfermion's MolecularData
    return molecule, n_qubits, n_orb, n_orb_occ, occ_indices_spin, \
        hamiltonian_fermOp, hamiltonian_qubitOp,energy_fci

def obtain_Hamiltonian(geometry: list,
                          basis: str = "sto-3g", 
                          spin: int = 0,
                          charge: int=0, 
                          with_fci: bool=True,
                          BK_reduce: bool=True):
    '''
    obtain the Hamiltoian based on the openfermion
    '''
#    molecule, n_qubits, n_orb, n_orb_occ, occ_indices_spin, \
#    hamiltonian_fermOp, hamiltonian_qubitOp,e\
#        = init_scf_old(geometry, with_fci, basis, spin,charge)
    molecule, n_qubits, n_orb, n_orb_occ, occ_indices_spin, \
               hamiltonian_fermOp, hamiltonian_qubitOp,e\
         = init_scf(geometry, basis, spin,
             freeze_indices_spatial=[],
             active_indices_spatial=[],
             run_fci = True,
             localized_orbitals= None,
             use_symmetry= False,
             override_symmetry_group= None,
             fermion_to_qubit_mapping= "jw")

    if BK_reduce:
        hamiltonian_qubitOp_reduced = openfermion.symmetry_conserving_bravyi_kitaev(hamiltonian_fermOp, int(n_orb) * 2, sum(molecule.nelec))
#        hamiltonian_qubitOp_reduced,nterms,ep_cont = chop_to_real(hamiltonian_qubitOp_reduced)
    else:
        hamiltonian_qubitOp_reduced,nterms,ep_cont = chop_to_real(hamiltonian_qubitOp)

    n_qubits=count_qubits(hamiltonian_qubitOp_reduced)

    ham_matrix = openfermion.get_sparse_operator(hamiltonian_qubitOp_reduced)
    e, eigvectors = scipy.sparse.linalg.eigsh(ham_matrix, k=1, which="SA")
    
#    print('The exact energy:',energy) 
#    sys.exit(0)

    print("n_qubits, n_orb, n_orb_occ, occ_indices_spin\n",n_qubits, n_orb, n_orb_occ, occ_indices_spin)
    return e, n_qubits, n_orb, n_orb_occ, occ_indices_spin, hamiltonian_qubitOp_reduced

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
