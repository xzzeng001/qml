import numpy
import openfermion


def _Pij(i: int, j: int):
    ia = i * 2 + 0
    ib = i * 2 + 1
    ja = j * 2 + 0
    jb = j * 2 + 1
    term1 = openfermion.FermionOperator(
        ((ja, 0), (ib, 0)),
        1.0
    )
    term2 = openfermion.FermionOperator(
        ((ia, 0), (jb, 0)),
        1.0
    )
    return numpy.sqrt(0.5) * (term1 + term2)


def _Pij_dagger(i: int, j: int):
    return openfermion.hermitian_conjugated(_Pij(i, j))


def _Qij_plus(i: int, j: int):
    ia = i * 2 + 0
    ib = i * 2 + 1
    ja = j * 2 + 0
    jb = j * 2 + 1
    term = openfermion.FermionOperator(
        ((ja, 0), (ia, 0)),
        1.0
    )
    return term


def _Qij_minus(i: int, j: int):
    ia = i * 2 + 0
    ib = i * 2 + 1
    ja = j * 2 + 0
    jb = j * 2 + 1
    term = openfermion.FermionOperator(
        ((jb, 0), (ib, 0)),
        1.0
    )
    return term


def _Qij_0(i: int, j: int):
    ia = i * 2 + 0
    ib = i * 2 + 1
    ja = j * 2 + 0
    jb = j * 2 + 1
    term1 = openfermion.FermionOperator(
        ((ja, 0), (ib, 0)),
        1.0
    )
    term2 = openfermion.FermionOperator(
        ((ia, 0), (jb, 0)),
        1.0
    )
    return numpy.sqrt(0.5) * (term1 - term2)


def _Qij_vec(i: int, j: int):
    return [_Qij_plus(i, j), _Qij_minus(i, j), _Qij_0(i, j)]


def _Qij_vec_dagger(i: int, j: int):
    return [openfermion.hermitian_conjugated(i) for i in _Qij_vec(i, j)]


def _Qij_vec_inner(a: int, b: int, i: int, j: int):
    vec_dagger = _Qij_vec_dagger(a, b)
    vec = _Qij_vec(i, j)
    return sum([vec[i] * vec_dagger[i] for i in range(len(vec))])


def spin_adapted_T1(i, j):
    """
    Spin-adapted single excitation operators.

    Args:
        i (int): index of the spatial orbital which the
            creation operator will act on.
        j (int): index of the spatial orbital which the
            annihilation operator will act on.

    Returns:
        tpq_list (list): Spin-adapted single excitation operators.

    Reference:
        Scuseria, G. E. et al., J. Chem. Phys. 89, 7382 (1988)
    """
    ia = i * 2 + 0
    ib = i * 2 + 1
    ja = j * 2 + 0
    jb = j * 2 + 1
    term1 = openfermion.FermionOperator(((ia, 1), (ja, 0)), 1.0)
    term2 = openfermion.FermionOperator(((ib, 1), (jb, 0)), 1.0)
    tpq_list = [term1 + term2]
    return tpq_list


def spin_adapted_T2(creation_list, annihilation_list):
    """
    Spin-adapted double excitation operators.

    Args:
        creation_list (list): list of spatial orbital indices which the
            creation operator will act on.
        annihilation_list (list): list of spatial orbital indices which the
            annihilation operator will act on.

    Returns:
        tpqrs_list (list): Spin-adapted double excitation operators.

    Reference:
        Igor O. Sokolov et al., J. Chem. Phys. 152, 124107 (2020)
        Ireneusz W. Bulik et al., J. Chem. Theory Comput. 11, 3171−3179 (2015)
        Scuseria, G. E. et al., J. Chem. Phys. 89, 7382 (1988)
    """
    p = creation_list[0]
    r = annihilation_list[0]
    q = creation_list[1]
    s = annihilation_list[1]
    tpqrs1 = _Pij_dagger(p, q) * _Pij(r, s)
    tpqrs2 = _Qij_vec_inner(p, q, r, s)
    tpqrs_list = [tpqrs1, tpqrs2]
    return tpqrs_list


def generate_molecule_uccsd(n_orb, n_orb_occ, anti_hermitian=True):
    """
    Generate UCCSD ansatz operator pool for molecular systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        anti_hermitian (bool): Whether to substract the hermitian conjugate.

    Returns:
        uccsd_operator_pool_fermOp (list):
            UCCSD Fermionic operators.
        uccsd_operator_pool_QubitOp (list):
            UCCSD Qubit operators under JW transformation.

    """

    n_orb_vir = n_orb - n_orb_occ
    occ_indices = [i for i in range(n_orb_occ)]
    vir_indices = [i + n_orb_occ for i in range(n_orb_vir)]

    T1_singles = []
    T2_doubles = []
    for p_idx in range(len(vir_indices)):
        p = vir_indices[p_idx]
        for q_idx in range(len(occ_indices)):
            q = occ_indices[q_idx]

            tpq_list = spin_adapted_T1(p, q)
            for idx in range(len(tpq_list)):
                tpq = tpq_list[idx]
                if anti_hermitian:
                    tpq = tpq - openfermion.hermitian_conjugated(tpq)
                tpq = openfermion.normal_ordered(tpq)
                if (tpq.many_body_order() > 0):
                    T1_singles.append(tpq)

    for p_idx in range(len(vir_indices)):
        p = vir_indices[p_idx]
        for q_idx in range(p_idx, len(vir_indices)):
            q = vir_indices[q_idx]
            for r_idx in range(len(occ_indices)):
                r = occ_indices[r_idx]
                for s_idx in range(r_idx, len(occ_indices)):
                    s = occ_indices[s_idx]

                    tpqrs_list = spin_adapted_T2([p, q], [r, s])
                    for idx in range(len(tpqrs_list)):
                        tpqrs = tpqrs_list[idx]
                        if anti_hermitian:
                            tpqrs = tpqrs - openfermion.hermitian_conjugated(tpqrs)
                        tpqrs = openfermion.normal_ordered(tpqrs)
                        if (tpqrs.many_body_order() > 0):
                            T2_doubles.append(tpqrs)

    uccsd_operator_pool_fermOp = T1_singles + T2_doubles
    uccsd_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in uccsd_operator_pool_fermOp]

    return uccsd_operator_pool_fermOp, uccsd_operator_pool_QubitOp


def generate_molecule_uccgsd(n_orb, n_orb_occ, anti_hermitian=True):
    """
    Generate UCCGSD ansatz operator pool for molecular systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        anti_hermitian (bool): Whether to substract the hermitian conjugate.

    Returns:
        uccsd_operator_pool_fermOp (list):
            UCCGSD Fermionic operators.
        uccsd_operator_pool_QubitOp (list):
            UCCGSD Qubit operators under JW transformation.
    """

    n_orb_vir = n_orb - n_orb_occ
    occ_indices = [i for i in range(n_orb)]
    vir_indices = [i for i in range(n_orb)]

    T1_singles = []
    T2_doubles = []
    for p_idx in range(len(vir_indices)):
        p = vir_indices[p_idx]
        for q_idx in range(len(occ_indices)):
            q = occ_indices[q_idx]

            tpq_list = spin_adapted_T1(p, q)
            for idx in range(len(tpq_list)):
                tpq = tpq_list[idx]
                if anti_hermitian:
                    tpq = tpq - openfermion.hermitian_conjugated(tpq)
                tpq = openfermion.normal_ordered(tpq)
                if (tpq.many_body_order() > 0):
                    T1_singles.append(tpq)

    pq = -1
    for p_idx in range(len(vir_indices)):
        p = vir_indices[p_idx]
        for q_idx in range(p_idx, len(vir_indices)):
            q = vir_indices[q_idx]
            pq += 1
            rs = -1
            for r_idx in range(len(occ_indices)):
                r = occ_indices[r_idx]
                for s_idx in range(r_idx, len(occ_indices)):
                    s = occ_indices[s_idx]
                    rs += 1
                    if (pq > rs):
                        continue

                    tpqrs_list = spin_adapted_T2([p, q], [r, s])
                    for idx in range(len(tpqrs_list)):
                        tpqrs = tpqrs_list[idx]
                        if anti_hermitian:
                            tpqrs = tpqrs - openfermion.hermitian_conjugated(tpqrs)
                        tpqrs = openfermion.normal_ordered(tpqrs)
                        if (tpqrs.many_body_order() > 0):
                            T2_doubles.append(tpqrs)

    uccsd_operator_pool_fermOp = T1_singles + T2_doubles
    uccsd_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in uccsd_operator_pool_fermOp]

    return uccsd_operator_pool_fermOp, uccsd_operator_pool_QubitOp


def _verify_kconserv(kpts,
                     k_idx_creation, k_idx_annihilation,
                     lattice_vec):
    """
    Helper function to check momentum conservation condition.

    Args:
        kpts (numpy.ndarray): Coordinates of k-points in reciprocal space.
        k_idx_creation (list): Indices of k-points which
            corresponds to creation operators.
        k_idx_annihilation (list): Indices of k-points which
            corresponds to annihilation operators.
        lattice_vec (numpy.ndarray): Lattice vectors.
    """
    sum_kpts = numpy.zeros(kpts[0].shape)
    for a in range(len(k_idx_creation)):
        sum_kpts += kpts[k_idx_creation[a]]
    for i in range(len(k_idx_annihilation)):
        sum_kpts -= kpts[k_idx_annihilation[i]]
    dots = numpy.dot(sum_kpts, lattice_vec / (2 * numpy.pi))
    """
    Every element in dots should be int if kconserv.
    """
    if ((abs(numpy.rint(dots) - dots) <= 1e-8).nonzero()[0].shape[0] == 3):
        return True
    return False


def generate_pbc_uccsd(n_orb, n_orb_occ,
                       kpts, m2k, lattice_vec,
                       complementary_pool=False,
                       anti_hermitian=True):
    """
    Generate UCCSD ansatz operator pool for periodic systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        kpts (numpy.ndarray): Coordinates of k-points.
        m2k: m2k returned by init_scf_pbc()
        lattice_vec (numpy.ndarray): Lattice vectors.
        complementary_pool (bool): Whether to add complementary terms
            in the operator pool.
        anti_hermitian (bool): Whether to substract the hermitian conjugate.

    Returns:
        uccsd_operator_pool_fermOp (list):
            UCCSD Fermionic operators.
        uccsd_operator_pool_QubitOp (list):
            UCCSD Qubit operators under JW transformation.
    """
    n_orb_vir = n_orb - n_orb_occ
    occ_indices = [i for i in range(n_orb_occ)]
    vir_indices = [i + n_orb_occ for i in range(n_orb_vir)]

    T1_singles = []
    T2_doubles = []
    # Complementary operator pool
    T1_singles_c = []
    T2_doubles_c = []
    for p_spatial in range(len(vir_indices)):
        p = vir_indices[p_spatial]
        kp_idx = m2k[p][0]
        for q_spatial in range(len(occ_indices)):
            q = occ_indices[q_spatial]
            kq_idx = m2k[q][0]

            if (_verify_kconserv(kpts, [kp_idx], [kq_idx],
                                 lattice_vec) is True):
                tpq_list = spin_adapted_T1(p, q)
                for idx in range(len(tpq_list)):
                    tpq = tpq_list[idx]
                    tpq_c = 1.j * (tpq + openfermion.hermitian_conjugated(tpq))
                    if anti_hermitian:
                        tpq = tpq - openfermion.hermitian_conjugated(tpq)
                    tpq = openfermion.normal_ordered(tpq)
                    tpq_c = openfermion.normal_ordered(tpq_c)
                    if (tpq.many_body_order() > 0):
                        T1_singles.append(tpq)
                    if (tpq_c.many_body_order() > 0):
                        T1_singles_c.append(tpq_c)

    for p_spatial in range(len(vir_indices)):
        p = vir_indices[p_spatial]
        kp_idx = m2k[p][0]
        for q_spatial in range(p_spatial, len(vir_indices)):
            q = vir_indices[q_spatial]
            kq_idx = m2k[q][0]
            for r_spatial in range(len(occ_indices)):
                r = occ_indices[r_spatial]
                kr_idx = m2k[r][0]
                for s_spatial in range(r_spatial, len(occ_indices)):
                    s = occ_indices[s_spatial]
                    ks_idx = m2k[s][0]

                    if (_verify_kconserv(kpts,
                                         [kp_idx, kq_idx], [kr_idx, ks_idx],
                                         lattice_vec) is True):
                        tpqrs_list = spin_adapted_T2([p, q], [r, s])
                        for idx in range(len(tpqrs_list)):
                            tpqrs = tpqrs_list[idx]
                            tpqrs_c = 1.j * (tpqrs + openfermion.hermitian_conjugated(tpqrs))
                            if anti_hermitian:
                                tpqrs = tpqrs - openfermion.hermitian_conjugated(tpqrs)
                            tpqrs = openfermion.normal_ordered(tpqrs)
                            tpqrs_c = openfermion.normal_ordered(tpqrs_c)
                            if (tpqrs.many_body_order() > 0):
                                T2_doubles.append(tpqrs)
                            if (tpqrs_c.many_body_order() > 0):
                                T2_doubles_c.append(tpqrs_c)

    uccsd_operator_pool_fermOp = T1_singles + T2_doubles
    if (complementary_pool is True):
        uccsd_operator_pool_fermOp += (T1_singles_c + T2_doubles_c)

    uccsd_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in uccsd_operator_pool_fermOp]

    return uccsd_operator_pool_fermOp, uccsd_operator_pool_QubitOp


def generate_pbc_uccgsd(n_orb, n_orb_occ,
                        kpts, m2k, lattice_vec,
                        complementary_pool=False,
                        anti_hermitian=True):
    """
    Generate UCCGSD ansatz operator pool for periodic systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        kpts (numpy.ndarray): Coordinates of k-points.
        m2k: m2k returned by init_scf_pbc()
        lattice_vec (numpy.ndarray): Lattice vectors.
        complementary_pool (bool): Whether to add complementary terms
            in the operator pool.
        anti_hermitian (bool): Whether to substract the hermitian conjugate.

    Returns:
        uccsd_operator_pool_fermOp (list):
            UCCSD Fermionic operators.
        uccsd_operator_pool_QubitOp (list):
            UCCSD Qubit operators under JW transformation.

    """
    n_orb_vir = n_orb - n_orb_occ
    occ_indices = [i for i in range(n_orb)]
    vir_indices = [i for i in range(n_orb)]

    T1_singles = []
    T2_doubles = []
    # Complementary operator pool
    T1_singles_c = []
    T2_doubles_c = []
    for p_spatial in range(len(vir_indices)):
        p = vir_indices[p_spatial]
        kp_idx = m2k[p][0]
        for q_spatial in range(len(occ_indices)):
            q = occ_indices[q_spatial]
            kq_idx = m2k[q][0]

            if (_verify_kconserv(kpts, [kp_idx], [kq_idx],
                                 lattice_vec) is True):
                tpq_list = spin_adapted_T1(p, q)
                for idx in range(len(tpq_list)):
                    tpq = tpq_list[idx]
                    tpq_c = 1.j * (tpq + openfermion.hermitian_conjugated(tpq))
                    if anti_hermitian:
                        tpq = tpq - openfermion.hermitian_conjugated(tpq)
                    tpq = openfermion.normal_ordered(tpq)
                    tpq_c = openfermion.normal_ordered(tpq_c)
                    if (tpq.many_body_order() > 0):
                        T1_singles.append(tpq)
                    if (tpq_c.many_body_order() > 0):
                        T1_singles_c.append(tpq_c)

    pq = -1
    for p_spatial in range(len(vir_indices)):
        p = vir_indices[p_spatial]
        kp_idx = m2k[p][0]
        for q_spatial in range(p_spatial, len(vir_indices)):
            q = vir_indices[q_spatial]
            kq_idx = m2k[q][0]
            pq += 1
            rs = -1
            for r_spatial in range(len(occ_indices)):
                r = occ_indices[r_spatial]
                kr_idx = m2k[r][0]
                for s_spatial in range(r_spatial, len(occ_indices)):
                    s = occ_indices[s_spatial]
                    ks_idx = m2k[s][0]
                    rs += 1
                    if (pq > rs):
                        continue

                    if (_verify_kconserv(kpts,
                                         [kp_idx, kq_idx], [kr_idx, ks_idx],
                                         lattice_vec) is True):
                        tpqrs_list = spin_adapted_T2([p, q], [r, s])
                        for idx in range(len(tpqrs_list)):
                            tpqrs = tpqrs_list[idx]
                            tpqrs_c = 1.j * (tpqrs + openfermion.hermitian_conjugated(tpqrs))
                            if anti_hermitian:
                                tpqrs = tpqrs - openfermion.hermitian_conjugated(tpqrs)
                            tpqrs = openfermion.normal_ordered(tpqrs)
                            tpqrs_c = openfermion.normal_ordered(tpqrs_c)
                            if (tpqrs.many_body_order() > 0):
                                T2_doubles.append(tpqrs)
                            if (tpqrs_c.many_body_order() > 0):
                                T2_doubles_c.append(tpqrs_c)

    uccsd_operator_pool_fermOp = T1_singles + T2_doubles
    if (complementary_pool is True):
        uccsd_operator_pool_fermOp += (T1_singles_c + T2_doubles_c)

    uccsd_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in uccsd_operator_pool_fermOp]

    return uccsd_operator_pool_fermOp, uccsd_operator_pool_QubitOp


def generate_molecule_eomip(n_orb, n_orb_occ, deexcitation=False):
    """
    Generate EOMIP operator pool for molecular systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        deexcitation (bool): Whether to include deexcitation operators.

    Returns:
        eomip_operator_pool_fermOp (list):
            EOMIP Fermionic operators.
        eomip_operator_pool_QubitOp (list):
            EOMIP Qubit operators under JW transformation.
    """
    n_orb_vir = n_orb - n_orb_occ

    IP1_singles = []
    IP2_doubles = []
    for i in range(n_orb_occ):
        ia = 2 * i
        ib = 2 * i + 1
        ri = openfermion.FermionOperator(
            ((ia, 0)),
            1.
        )
        IP1_singles.append(ri)
        if deexcitation:
            IP1_singles.append(openfermion.hermitian_conjugated(ri))

    for i in range(n_orb_occ):
        ia = 2 * i
        ib = 2 * i + 1
        for j in range(n_orb_occ):
            ja = 2 * j
            jb = 2 * j + 1
            for b in range(n_orb_vir):
                ba = 2 * n_orb_occ + 2 * b
                bb = 2 * n_orb_occ + 2 * b + 1
                rbji = openfermion.FermionOperator(
                    ((ba, 1), (ja, 0), (ia, 0)),
                    1. / 2.
                )
                rbji += openfermion.FermionOperator(
                    ((bb, 1), (jb, 0), (ia, 0)),
                    1. / 2.
                )
                IP2_doubles.append(rbji)
                if deexcitation:
                    IP2_doubles.append(openfermion.hermitian_conjugated(rbji))

    eomip_operator_pool_fermOp = IP1_singles + IP2_doubles
    eomip_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in eomip_operator_pool_fermOp]
    return eomip_operator_pool_fermOp, eomip_operator_pool_QubitOp


def generate_molecule_eomea(n_orb, n_orb_occ, deexcitation=False):
    """
    Generate EOMEA operator pool for molecular systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        deexcitation (bool): Whether to include deexcitation operators.

    Returns:
        eomea_operator_pool_fermOp (list):
            EOMEA Fermionic operators.
        eomea_operator_pool_QubitOp (list):
            EOMEA Qubit operators under JW transformation.
    """
    n_orb_vir = n_orb - n_orb_occ
    EA1_singles = []
    EA2_doubles = []
    for a in range(n_orb_vir):
        aa = 2 * n_orb_occ + 2 * a
        ab = 2 * n_orb_occ + 2 * a + 1
        ra = openfermion.FermionOperator(
            ((aa, 1)),
            1.
        )
        EA1_singles.append(ra)
        if deexcitation:
            EA1_singles.append(openfermion.hermitian_conjugated(ra))

    for a in range(n_orb_vir):
        aa = 2 * n_orb_occ + 2 * a
        ab = 2 * n_orb_occ + 2 * a + 1
        for b in range(n_orb_vir):
            ba = 2 * n_orb_occ + 2 * b
            bb = 2 * n_orb_occ + 2 * b + 1
            for j in range(n_orb_occ):
                ja = 2 * j
                jb = 2 * j + 1
                rabj = openfermion.FermionOperator(
                    ((aa, 1), (ba, 1), (ja, 0)),
                    1. / 2.
                )
                rabj += openfermion.FermionOperator(
                    ((aa, 1), (bb, 1), (jb, 0)),
                    1. / 2.
                )
                EA2_doubles.append(rabj)
                if deexcitation:
                    EA2_doubles.append(openfermion.hermitian_conjugated(rabj))

    eomea_operator_pool_fermOp = EA1_singles + EA2_doubles
    eomea_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in eomea_operator_pool_fermOp]
    return eomea_operator_pool_fermOp, eomea_operator_pool_QubitOp


def generate_molecule_eomee(n_orb, n_orb_occ, deexcitation=False):
    """
    Generate EOMEE operator pool for molecular systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        deexcitation (bool): Whether to include deexcitation operators.

    Returns:
        eomee_operator_pool_fermOp (list):
            EOMEE Fermionic operators.
        eomee_operator_pool_QubitOp (list):
            EOMEE Qubit operators under JW transformation.
    """
    eomee_operator_pool_fermOp_, eomee_operator_pool_QubitOp_ = \
        generate_molecule_uccsd(n_orb, n_orb_occ, anti_hermitian=False)
    n_terms = len(eomee_operator_pool_fermOp_)
    eomee_operator_pool_fermOp = []
    eomee_operator_pool_QubitOp = []
    for i in range(n_terms):
        fermOp_i = eomee_operator_pool_fermOp_[i]
        qubitOp_i = eomee_operator_pool_QubitOp_[i]
        eomee_operator_pool_fermOp.append(fermOp_i)
        eomee_operator_pool_QubitOp.append(qubitOp_i)
        if deexcitation:
            eomee_operator_pool_fermOp.append(
                openfermion.hermitian_conjugated(fermOp_i))
            eomee_operator_pool_QubitOp.append(
                openfermion.hermitian_conjugated(qubitOp_i))
    return eomee_operator_pool_fermOp, eomee_operator_pool_QubitOp


def generate_molecule_eomee_unrestricted(n_orb, n_orb_occ, deexcitation=False):
    """
    Generate EOMEE operator pool for molecular systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        deexcitation (bool): Whether to include deexcitation operators.

    Returns:
        eomee_operator_pool_fermOp (list):
            EOMEE Fermionic operators.
        eomee_operator_pool_QubitOp (list):
            EOMEE Qubit operators under JW transformation.
    """
    n_qubits = n_orb * 2
    occ_indices_spin = [i for i in range(2 * n_orb_occ)]
    vir_indices_spin = [i + 2 * n_orb_occ
                        for i in range(n_qubits - 2 * n_orb_occ)]
    EE1_singles = []
    EE2_doubles = []
    for a in vir_indices_spin:
        for i in occ_indices_spin:
            rai = openfermion.FermionOperator(
                ((a, 1), (i, 0)),
                1.
            )
            rai = openfermion.normal_ordered(rai)
            if rai.many_body_order() > 0:
                EE1_singles.append(rai)
                if deexcitation:
                    EE1_singles.append(openfermion.hermitian_conjugated(rai))

    for a in vir_indices_spin:
        for b in vir_indices_spin:
            if vir_indices_spin.index(b) <= vir_indices_spin.index(a):
                continue
            for i in occ_indices_spin:
                for j in occ_indices_spin:
                    if occ_indices_spin.index(j) <= occ_indices_spin.index(i):
                        continue
                    rabij = openfermion.FermionOperator(
                        ((a, 1), (b, 1), (i, 0), (j, 0)),
                        1.
                    )
                    rabij = openfermion.normal_ordered(rabij)
                    if rabij.many_body_order() > 0:
                        EE2_doubles.append(rabij)
                        if deexcitation:
                            EE2_doubles.append(
                                openfermion.hermitian_conjugated(rabij))

    eomee_operator_pool_fermOp = EE1_singles + EE2_doubles
    eomee_operator_pool_QubitOp = [openfermion.jordan_wigner(i)
                                   for i in eomee_operator_pool_fermOp]
    return eomee_operator_pool_fermOp, eomee_operator_pool_QubitOp


def generate_pbc_eomip(n_orb, n_orb_occ,
                       kpts, m2k, lattice_vec,
                       deexcitation=False):
    """
    Generate EOMIP operator pool for periodic systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        kpts (numpy.ndarray): Coordinates of k-points.
        m2k: m2k returned by init_scf_pbc()
        lattice_vec (numpy.ndarray): Lattice vectors.
        deexcitation (bool): Whether to include deexcitation operators.

    Returns:
        eomip_operator_pool_fermOp (list):
            EOMIP Fermionic operators.
        eomip_operator_pool_QubitOp (list):
            EOMIP Qubit operators under JW transformation.
    """
    kshift = 0
    n_orb_vir = n_orb - n_orb_occ

    IP1_singles = []
    IP2_doubles = []
    for i in range(n_orb_occ):
        ki_idx = m2k[i][0]
        if (ki_idx != kshift):
            continue
        ia = 2 * i
        ib = 2 * i + 1
        ri = openfermion.FermionOperator(
            ((ia, 0)),
            1.
        )
        IP1_singles.append(ri)
        if deexcitation:
            IP1_singles.append(openfermion.hermitian_conjugated(ri))

    for i in range(n_orb_occ):
        ki_idx = m2k[i][0]
        ia = 2 * i
        ib = 2 * i + 1
        for j in range(n_orb_occ):
            kj_idx = m2k[j][0]
            ja = 2 * j
            jb = 2 * j + 1
            for b in range(n_orb_vir):
                kb_idx = m2k[n_orb_occ + b][0]
                ba = 2 * (n_orb_occ + b)
                bb = 2 * (n_orb_occ + b) + 1
                if (_verify_kconserv(
                        kpts, [kb_idx, kshift], [ki_idx, kj_idx],
                        lattice_vec) is True):
                    rbji = openfermion.FermionOperator(
                        ((ba, 1), (ja, 0), (ia, 0)),
                        1. / 2.
                    )
                    rbji += openfermion.FermionOperator(
                        ((bb, 1), (jb, 0), (ia, 0)),
                        1. / 2.
                    )
                    IP2_doubles.append(rbji)
                    if deexcitation:
                        IP2_doubles.append(
                            openfermion.hermitian_conjugated(rbji))

    eomip_operator_pool_fermOp = IP1_singles + IP2_doubles
    eomip_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in eomip_operator_pool_fermOp]
    return eomip_operator_pool_fermOp, eomip_operator_pool_QubitOp


def generate_pbc_eomea(n_orb, n_orb_occ,
                       kpts, m2k, lattice_vec,
                       deexcitation=False):
    """
    Generate EOMEA operator pool for periodic systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        kpts (numpy.ndarray): Coordinates of k-points.
        m2k: m2k returned by init_scf_pbc()
        lattice_vec (numpy.ndarray): Lattice vectors.
        deexcitation (bool): Whether to include deexcitation operators.

    Returns:
        eomea_operator_pool_fermOp (list):
            EOMEA Fermionic operators.
        eomea_operator_pool_QubitOp (list):
            EOMEA Qubit operators under JW transformation.
    """
    kshift = 0
    n_orb_vir = n_orb - n_orb_occ

    EA1_singles = []
    EA2_doubles = []
    for a in range(n_orb_vir):
        ka_idx = m2k[n_orb_occ + a][0]
        if (ka_idx != kshift):
            continue
        aa = 2 * (n_orb_occ + a)
        ab = 2 * (n_orb_occ + a) + 1
        ra = openfermion.FermionOperator(
            ((aa, 1)),
            1.
        )
        EA1_singles.append(ra)
        if deexcitation:
            EA1_singles.append(openfermion.hermitian_conjugated(ra))

    for a in range(n_orb_vir):
        ka_idx = m2k[n_orb_occ + a][0]
        aa = 2 * (n_orb_occ + a)
        ab = 2 * (n_orb_occ + a) + 1
        for b in range(n_orb_vir):
            kb_idx = m2k[n_orb_occ + b][0]
            ba = 2 * (n_orb_occ + b)
            bb = 2 * (n_orb_occ + b) + 1
            for j in range(n_orb_occ):
                kj_idx = m2k[j][0]
                ja = 2 * j
                jb = 2 * j + 1
                if (_verify_kconserv(
                    kpts, [ka_idx, kb_idx], [kj_idx, kshift],
                        lattice_vec) is True):
                    rabj = openfermion.FermionOperator(
                        ((aa, 1), (ba, 1), (ja, 0)),
                        1. / 2.
                    )
                    rabj += openfermion.FermionOperator(
                        ((aa, 1), (bb, 1), (jb, 0)),
                        1. / 2.
                    )
                    EA2_doubles.append(rabj)
                    if deexcitation:
                        EA2_doubles.append(
                            openfermion.hermitian_conjugated(rabj))

    eomea_operator_pool_fermOp = EA1_singles + EA2_doubles
    eomea_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in eomea_operator_pool_fermOp]
    return eomea_operator_pool_fermOp, eomea_operator_pool_QubitOp


def qubit_adapt_pool(n: int):
    operator_pool = []
    assert(n >= 3)
    if (n == 3):
        operator_pool.append(
            openfermion.QubitOperator(((2, "Z"), (1, "Z"), (0, "Y"),), 1.)
        )
        operator_pool.append(
            openfermion.QubitOperator(((2, "Z"), (1, "Y"),), 1.)
        )
        operator_pool.append(
            openfermion.QubitOperator(((2, "Y"),), 1.)
        )
        operator_pool.append(
            openfermion.QubitOperator(((1, "Y"),), 1.)
        )
    else:
        operator_pool_ = qubit_adapt_pool(n - 1)
        for i in operator_pool_:
            operator_pool.append(
                openfermion.QubitOperator(((n - 1, "Z"),), 1.) * i
            )
        operator_pool.append(
            openfermion.QubitOperator(((n - 1, "Y"),), 1.)
        )
        operator_pool.append(
            openfermion.QubitOperator(((n - 2, "Y"),), 1.)
        )
    return operator_pool


def convert_to_qubit_adaptvqe_pool(operator_pool_qubitOp: list):
    qubit_adaptvqe_operator_pool = []
    for i in range(len(operator_pool_qubitOp)):
        qubitOp_i = operator_pool_qubitOp[i]
        for term, coeff in qubitOp_i.terms.items():
            term_list = list(term)
            term_list_new = [i for i in term_list if i[1] != "Z"]
            term_new = tuple(term_list_new)
            y_count = 0
            for term_new_term in term_new:
                if term_new_term[1] == "Y":
                    y_count += 1
            if y_count % 2 == 0:
                continue
            qubit_adaptvqe_operator_pool.append(
                openfermion.QubitOperator(term_new, coeff)
            )
    return qubit_adaptvqe_operator_pool


def _qubit_excit(i: int):
    qubitOp1 = openfermion.QubitOperator(
        ((i, "X"),), 1.
    )
    qubitOp2 = openfermion.QubitOperator(
        ((i, "Y"),), 1.j
    )
    qubitOp = (qubitOp1 - qubitOp2) * 0.5
    return qubitOp


def _qubit_deexcit(i: int):
    qubitOp1 = openfermion.QubitOperator(
        ((i, "X"),), 1.
    )
    qubitOp2 = openfermion.QubitOperator(
        ((i, "Y"),), 1.j
    )
    qubitOp = (qubitOp1 + qubitOp2) * 0.5
    return qubitOp


def QubitExcitationOperator(term: tuple, coeff: complex):
    """
    Create Qubit excitation operators. Takes the same arguments as
        openfermion's FermionOperator.

    Definition:
        Q^+_n = 1/2 (X_n - iY_n)
        Q_n = 1/2 (X_n + iY_n)

    Returns:
        qubitOp (openfermion.QubitOperator): Qubit excitation operators
            represented by openfermion's QubitOperator

    Examples:
        >>> import openfermion
        >>> from utils import QubitExcitationOperator
        >>> qubitEOp = QubitExcitationOperator(((2, 0), (0, 1)), 2.j)
        >>> fermOp = openfermion.FermionOperator(((2, 0), (0, 1)), 2.j)
        >>> qubitOp = openfermion.jordan_wigner(fermOp)
        >>> qubitEOp
        0.5j [X0 X2] +
        (-0.5+0j) [X0 Y2] +
        (0.5+0j) [Y0 X2] +
        0.5j [Y0 Y2]
        >>> qubitOp
        -0.5j [X0 Z1 X2] +
        (0.5+0j) [X0 Z1 Y2] +
        (-0.5+0j) [Y0 Z1 X2] +
        -0.5j [Y0 Z1 Y2]
    """
    qubitEOp = openfermion.QubitOperator((), 1.)
    for term_i in term:
        qubitEOp_i = _qubit_excit(term_i[0]) if term_i[1] == 1 \
            else _qubit_deexcit(term_i[0])
        qubitEOp *= qubitEOp_i
    qubitEOp *= coeff
    return qubitEOp


def convert_fermOp_to_qubitEOp(fermOp_list):
    if (isinstance(fermOp_list, openfermion.FermionOperator)):
        qubitEOp = openfermion.QubitOperator()
        for term, coeff in fermOp_list.terms.items():
            qubitEOp += \
                QubitExcitationOperator(
                    term,
                    coeff
                )
        return qubitEOp
    assert(isinstance(fermOp_list, list))
    qubitEOp_list = []
    for fermOp_i in fermOp_list:
        qubitEOp_i = openfermion.QubitOperator()
        for term, coeff in fermOp_i.terms.items():
            qubitEOp_i += \
                QubitExcitationOperator(
                    term,
                    coeff
                )
        qubitEOp_list.append(qubitEOp_i)
    return qubitEOp_list


def geigh(H: numpy.ndarray, S: numpy.ndarray, sort_eigs=False):
    """
    Solve the generalized eigenvalue problem HC=SCE
    """
    import scipy
    if (numpy.linalg.norm(H - H.T.conj()) > 1e-9):
        print("WARNING: H not hermitian !")
        print("Norm of H - H.T.conj(): %20.16f" %
              (numpy.linalg.norm(H - H.T.conj())))
        print("AbsMax of H - H.T.conj(): %20.16f" %
              (numpy.abs(H - H.T.conj()).max()))
    if (numpy.linalg.norm(S - S.T.conj()) > 1e-9):
        print("WARNING: S not hermitian !")
        print("Norm of S - S.T.conj(): %20.16f" %
              (numpy.linalg.norm(S - S.T.conj())))
        print("AbsMax of S - S.T.conj(): %20.16f" %
              (numpy.abs(S - S.T.conj()).max()))
    D = None
    V = None
    try:
        D, V = scipy.linalg.eigh(H, S)
    except numpy.linalg.LinAlgError:
        try:
            print("WARNING: scipy's built in eigh() failed. Try using SVD.")
            U, S0, Vh = numpy.linalg.svd(S)
            U1 = U[:, numpy.abs(S0) > 0.0]
            T = (U1.T.conj()).dot(H.dot(U1))
            G = numpy.diag(S0[: U1.shape[1]])
            D, V = scipy.linalg.eig(T, G)
            V = U1.dot(V.dot(U1.T.conj()))
        except numpy.linalg.LinAlgError:
            print("SVD failed. Using bare eig(). The calculated \
eigenvalues and eigenvectors may have duplicates!")
            D, V = scipy.linalg.eig(H, S)
        else:
            pass
    else:
        pass
    V = V / numpy.linalg.norm(V, axis=0)
    if (sort_eigs is True):
        idx_sorted = D.real.argsort()
        return D[idx_sorted], V[idx_sorted]
    else:
        return D, V


def decompose_trottered_qubitOp(qubitOp: openfermion.QubitOperator,
                                coeff_idx: int):
    """
    Simulate exp(i * (qubitOp / i) * coeff). coeff is some external coefficient.
    qubitOp may contain multiple terms. The i is contained in the qubitOp.

    Args:
        qubitOp (openfermion.QubitOperator): The qubit operator which
            is put on the exponential term.
        coeff_idx: The index of the coefficient.

    Returns:
        gates_apply (list): A list of gates which is to apply on the qubits.
            The order is:
            gates_apply[N]...gates_apply[1] gates_apply[0] |psi>

    Notes:
        HY = 2 ** 0.5 / 2. * numpy.array(
            [[1.0, -1.j],
             [1.j, -1.0]])

    Notes:
        The element of the returned gates_apply is tuples. There are three
        types:
        1: ("H", idx) or ("HY", idx):
            "H" or "HY": The H gate or HY gate.
            idx: The index of the qubit on which the gate will apply.
        2: ("CNOT", (ctrl_idx, apply_idx)):
            "CNOT": The CNOT gate.
            ctrl_idx: The control qubit.
            apply_idx: The active qubit.
        3: ("RZ", (idx, coeff, coeff_idx)):
            "RZ": The RZ gate.
            idx: The index of the qubit on which the gate will apply.
            coeff: The preceding factor.
            coeff_idx: The index of the external coefficient.

    Notes:
        The coeff_idx is an indicator of parameters. For example, we want to
        obtain the Trottered parametric circuit for P = c1 P1 + c2 P2 where Pi
        is a Pauli string, then we can call decompose_trottered_qubitOp(P1, idx1)
        and decompose_trottered_qubitOp(P2, idx2), assuming the indices for c1 and
        c2 are idx1 and idx2 respectively in a list C. Then, the coefficient
        of the RZ gate in exp(c1*P1) is equal to the preceding factor * C[idx1],
        where C[idx1] = c1

    Examples:
        >>> from utils import decompose_trottered_qubitOp
        >>> from openfermion import QubitOperator
        >>> qubitOp = QubitOperator("X0 Y1", 2.j) + QubitOperator("Z0 X1", -1.j)
        >>> qubitOp
        2j [X0 Y1] +
        -1j [Z0 X1]
        >>> coeffs = [1.]
        >>> gates_list = decompose_trottered_qubitOp(qubitOp, 0)
        >>> gates_list
        [('H', 0), ('HY', 1), ('CNOT', (1, 0)), ('RZ', (0, (-4+0j), 0)), ('CNOT', (1, 0)), ('H', 0), ('HY', 1), ('H', 1), ('CNOT', (1, 0)), ('RZ', (0, (2+0j), 0)), ('CNOT', (1, 0)), ('H', 1)]
    """

    gates_apply_dict = {
        "X": "H",
        "Y": "HY"
    }
    gates_apply = []
    qubitOp_list = list(qubitOp)
    for i in range(len(qubitOp_list)):
        qubitOp_i = qubitOp_list[i]
        terms_i = list(qubitOp_i.terms.keys())[0]
        coeff_i = qubitOp_i.terms[terms_i]
        assert(type(coeff_i) is complex)
        assert(numpy.isclose(coeff_i.real, 0.))
        if len(terms_i) == 0:
            """
            We ignore global phase.
            """
            continue
        parameter_i = coeff_i / 1.j
        idx_list_i = []
        qubit_gate_i = []
        for idx_apply, qubit_gate_apply in terms_i:
            idx_list_i.append(idx_apply)
            qubit_gate_i.append(qubit_gate_apply)

        for idx_op in range(len(idx_list_i)):
            if qubit_gate_i[idx_op] in gates_apply_dict:
                gates_apply.append(
                    (gates_apply_dict[qubit_gate_i[idx_op]], idx_list_i[idx_op]))
        for idx_op in reversed(range(len(idx_list_i) - 1)):
            gates_apply.append(
                ("CNOT", (idx_list_i[idx_op + 1], idx_list_i[idx_op]))
            )
        gates_apply.append(
            ("RZ", (idx_list_i[0], - 2 * parameter_i, coeff_idx))
        )
        for idx_op in range(len(idx_list_i) - 1):
            gates_apply.append(
                ("CNOT", (idx_list_i[idx_op + 1], idx_list_i[idx_op]))
            )
        for idx_op in range(len(idx_list_i)):
            if qubit_gate_i[idx_op] in gates_apply_dict:
                gates_apply.append(
                    (gates_apply_dict[qubit_gate_i[idx_op]], idx_list_i[idx_op]))
    return gates_apply


def parity_transformation(fermOp: openfermion.FermionOperator,
                          n_qubits: int,
                          taper_two_qubits: bool = False):
    fermOp_ = openfermion.reorder(fermOp, openfermion.up_then_down, n_qubits)
    result = openfermion.binary_code_transform(
        fermOp_, openfermion.parity_code(n_qubits))
    if (taper_two_qubits is False):
        return result
    else:
        qubitOp_remove_two = openfermion.QubitOperator()
        fermOp_remove_two = openfermion.FermionOperator()
        openfermion.symmetry_conserving_bravyi_kitaev
        remove_idx0 = 0  # n_qubits // 2 - 1
        remove_idx1 = n_qubits // 2 - 1  # n_qubits - 1
        for term, coeff in result.terms.items():
            new_term = []
            for i in term:
                # if i[0] < remove_idx0:
                #     new_term.append(i)
                # elif i[0] > remove_idx0 and i[0] < remove_idx1:
                #     new_term.append((i[0] - 1, i[1]))
                # else:
                #     pass
                if i[0] < remove_idx1:
                    new_term.append((i[0] - 1, i[1]))
                else:
                    new_term.append((i[0] - 2, i[1]))
            qubitOp_remove_two += openfermion.QubitOperator(
                tuple(new_term), coeff
            )
        return qubitOp_remove_two


def tapering_two(fermOp, n_qubits, n_electrons):
    fermOp_reorder = openfermion.reorder(
        fermOp, openfermion.up_then_down, n_qubits)
    checksum = openfermion.checksum_code(
        n_qubits // 2, (n_electrons // 2) % 2) * 2
    qubitOp = openfermion.binary_code_transform(fermOp_reorder, checksum)
    return qubitOp
