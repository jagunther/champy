import pytest
import numpy as np
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


@pytest.mark.parametrize("hamil_random", [(8, 8)], indirect=True)
def test_jordan_wigner(hamil_random):
    """Hamiltonian with constant and 1-electron terms only: h2e=0."""
    elstruc = hamil_random
    n = elstruc.num_orb
    pauli_hamil = elstruc.to_MajoranaPair().jordan_wigner()
    valid, error = _f2q_valid(elstruc=elstruc, pauli_hamil=pauli_hamil)
    assert valid
