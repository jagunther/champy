import pytest
import numpy as np
from unittest.mock import patch
from champy.ElectronicStructure import ElectronicStructure
from champy.PauliHamiltonian import PauliHamiltonian


def _f2q_valid(elstruc: ElectronicStructure, pauli_hamil: PauliHamiltonian):
    spec_elstruc = np.linalg.eigvals(elstruc.to_sparse_matrix()) + elstruc.constant
    spec_pauli = np.real(
        (
            np.linalg.eigvals(pauli_hamil.to_sparse_matrix().toarray())
            + pauli_hamil.constant
        )
    )
    print(spec_elstruc)
    print(spec_pauli)

    # check if electronic spectrum is subset of pauli spectrum
    for e in spec_elstruc:
        if not np.any(np.isclose(spec_pauli, e, atol=1e-6)):
            return False, e
    return True, None


@pytest.mark.parametrize("hamil_random", [(4, 4)], indirect=True)
def test_plot_orbital_graph(hamil_random):
    """plot_orbital_graph() runs without error on a random Hamiltonian."""
    majorana = hamil_random.to_MajoranaPair()
    with patch("matplotlib.pyplot.show"):
        result = majorana.plot_orbital_graph()
    assert result is None


@pytest.mark.parametrize("hamil_random", [(4, 4)], indirect=True)
def test_jw_matrix(hamil_random):
    """jordan_wigner() via jw_matrix(): FCI spectrum is subset of Pauli spectrum."""
    elstruc = hamil_random
    pauli_hamil = elstruc.to_MajoranaPair().jordan_wigner()
    valid, error = _f2q_valid(elstruc=elstruc, pauli_hamil=pauli_hamil)
    assert valid


def test_jordan_wigner_vs_openfermion(rhf_h2o):
    """jordan_wigner() spectrum agrees with openfermion's JW on H2O/cc-pVDZ active space.

    Both sides are built from the same champy integrals (h0, h1e, h2e), so the
    comparison is independent of active-space selection conventions.
    Spin-orbital ordering: alternating (P = 2p+σ, σ∈{up,down}).
    """
    import openfermion as of

    num_orb, num_elec = 4, 4
    hamil = ElectronicStructure.from_pyscf(rhf_h2o, num_orb=num_orb, num_elec=num_elec)
    n = hamil.num_orb
    N = 2 * n

    # champy: H = h0 + Σ_{pq,σ} h1e[p,q] a†_{pσ} a_{qσ}
    #              + 1/2 Σ_{pqrs,στ} h2e[p,q,r,s] a†_{pσ} a†_{rτ} a_{sτ} a_{qσ}
    # OF InteractionOperator: H = const + Σ_{PQ} T[P,Q] a†_P a_Q
    #                              + Σ_{PQRS} V[P,Q,R,S] a†_P a†_Q a_R a_S
    # with P = 2p+σ:  T[2p+σ,2q+σ] = h1e[p,q]
    #                 V[2p+σ, 2r+τ, 2s+τ, 2q+σ] = 1/2 * h2e[p,q,r,s]
    T = np.zeros((N, N))
    for p in range(n):
        for q in range(n):
            T[2 * p, 2 * q] = hamil.h1e[p, q]
            T[2 * p + 1, 2 * q + 1] = hamil.h1e[p, q]

    V = np.zeros((N, N, N, N))
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    for sig in range(2):
                        for tau in range(2):
                            V[2*p+sig, 2*r+tau, 2*s+tau, 2*q+sig] += 0.5 * hamil.h2e[p, q, r, s]

    of_op = of.InteractionOperator(constant=hamil.h0, one_body_tensor=T, two_body_tensor=V)
    qop = of.jordan_wigner(of_op)
    mat_of = of.get_sparse_operator(qop, n_qubits=N).toarray()
    spec_of = np.sort(np.real(np.linalg.eigvals(mat_of)))

    pauli_hamil = hamil.to_MajoranaPair().jordan_wigner()
    mat_champy = pauli_hamil.to_sparse_matrix().toarray() + pauli_hamil.constant * np.eye(2**N)
    spec_champy = np.sort(np.real(np.linalg.eigvals(mat_champy)))

    assert np.allclose(spec_champy, spec_of, atol=1e-6)
