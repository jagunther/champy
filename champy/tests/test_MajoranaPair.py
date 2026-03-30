import pytest
import numpy as np
from champy.ElectronicStructure import ElectronicStructure


def test_f2q_constant_only():
    """Hamiltonian with only a constant: h1e=0, h2e=0."""
    n, num_elec = 2, 2
    h0 = 3.7
    h1e = np.zeros((n, n))
    h2e = np.zeros((n, n, n, n))
    elec = ElectronicStructure(h0, h1e, h2e, num_elec)

    elec_eigs = _spectrum_elec(elec)
    pauli_eigs = _spectrum_pauli(elec.to_MajoranaPair().jordan_wigner())

    ok, missing = _elec_spectrum_is_subset(elec_eigs, pauli_eigs)
    assert ok, f"eigenvalue {missing} not found in Pauli spectrum"


def _spectrum_elec(elec):
    mat = elec.to_sparse_matrix()
    eigs = np.linalg.eigvalsh(mat)
    return np.sort(eigs + elec.constant)


def _spectrum_pauli(pauli):
    mat = pauli.to_sparse_matrix().toarray() + pauli.constant * np.eye(pauli.dimension)
    return np.sort(np.linalg.eigvalsh(np.real_if_close(mat)))


def _elec_spectrum_is_subset(elec_eigs, pauli_eigs, atol=1e-6):
    """Check every eigenvalue of ElectronicStructure appears in PauliHamiltonian spectrum."""
    for e in elec_eigs:
        if not np.any(np.isclose(pauli_eigs, e, atol=atol)):
            return False, e
    return True, None


@pytest.mark.parametrize("hamil_random", [(2, 2)], indirect=True)
def test_f2q_constant_and_1e(hamil_random):
    """Hamiltonian with constant and 1-electron terms only: h2e=0."""
    # h1e = np.random.rand(n, 2)
    # h1e = h1e + h1e.T
    # h2e = np.zeros((n, n, n, n))
    elstruc = hamil_random
    elstruc.h2e = np.zeros((2, 2, 2, 2))
    pauli_hamil = elstruc.to_MajoranaPair().jordan_wigner()
    spec_elstruc = np.linalg.eigvals(elstruc.to_sparse_matrix()) + elstruc.constant
    spec_pauli = np.real(
        (
            np.linalg.eigvals(pauli_hamil.to_sparse_matrix().toarray())
            + pauli_hamil.constant
        )
    )
    res, error = _elec_spectrum_is_subset(spec_elstruc, spec_pauli)
    assert res, print(error)


def test_f2q_spectrum(hamil_random):
    elec = hamil_random
    elec_eigs = _spectrum_elec(elec)
    pauli_eigs = _spectrum_pauli(elec.to_MajoranaPair().f2q())

    ok, missing = _elec_spectrum_is_subset(elec_eigs, pauli_eigs)
    assert ok, f"eigenvalue {missing} not found in Pauli spectrum"
