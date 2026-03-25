import numpy as np
import pytest

from champy.PauliHamiltonian import PauliHamiltonian

PARAMS = [(2, 5), (3, 8), (4, 15)]


@pytest.mark.parametrize("random_pauli_hamiltonian", PARAMS, indirect=True)
def test_dimension(random_pauli_hamiltonian):
    hamil, num_qubits, _ = random_pauli_hamiltonian
    assert hamil.dimension == 2**num_qubits


@pytest.mark.parametrize("random_pauli_hamiltonian", PARAMS, indirect=True)
def test_constant_extracted_from_identity(random_pauli_hamiltonian):
    hamil, num_qubits, _ = random_pauli_hamiltonian
    identity_label = "I" * num_qubits
    assert identity_label not in hamil.paulis
    assert hamil.constant != 0.0


@pytest.mark.parametrize("random_pauli_hamiltonian", PARAMS, indirect=True)
def test_paulis_and_coeffs_shape(random_pauli_hamiltonian):
    hamil, _, num_terms = random_pauli_hamiltonian
    assert hamil.paulis.shape == hamil.coeffs.shape
    assert len(hamil.paulis) == num_terms - 1  # exactly one identity term was extracted


@pytest.mark.parametrize("random_pauli_hamiltonian", PARAMS, indirect=True)
def test_to_sparse_matrix(random_pauli_hamiltonian):
    hamil, _, _ = random_pauli_hamiltonian
    assert np.allclose(hamil.to_sparse_matrix().toarray(), hamil._dense_matrix())


@pytest.mark.parametrize("random_pauli_hamiltonian", PARAMS, indirect=True)
def test_addition(random_pauli_hamiltonian):
    "takes random Hamiltonian H, splits into H1, H2 and checks whether H = H1 + H2"
    hamil, _, _ = random_pauli_hamiltonian
    n = len(hamil.paulis) // 2
    labels_a, weights_a = hamil.paulis[:n].tolist(), hamil.coeffs[:n].tolist()
    labels_b, weights_b = hamil.paulis[n:].tolist(), hamil.coeffs[n:].tolist()
    hamil_a = PauliHamiltonian.from_labels_and_weights(labels_a, weights_a)
    hamil_b = PauliHamiltonian.from_labels_and_weights(labels_b, weights_b)
    result = hamil_a + hamil_b
    expected = PauliHamiltonian.from_labels_and_weights(
        labels_a + labels_b, weights_a + weights_b
    )
    assert np.allclose(
        result.to_sparse_matrix().toarray(), expected.to_sparse_matrix().toarray()
    )
    assert np.isclose(result.constant, expected.constant)


@pytest.mark.parametrize("random_pauli_hamiltonian", PARAMS, indirect=True)
def test_subtraction(random_pauli_hamiltonian):
    "takes random Hamiltonian H, splits into H1, H2 and checks whether H = H1 - (-H2)"
    hamil, _, _ = random_pauli_hamiltonian
    n = len(hamil.paulis) // 2
    labels_a, weights_a = hamil.paulis[:n].tolist(), hamil.coeffs[:n].tolist()
    labels_b, weights_b = hamil.paulis[n:].tolist(), hamil.coeffs[n:].tolist()
    hamil_a = PauliHamiltonian.from_labels_and_weights(labels_a, weights_a)
    hamil_b = PauliHamiltonian.from_labels_and_weights(labels_b, weights_b)
    result = hamil_a - hamil_b
    expected = PauliHamiltonian.from_labels_and_weights(
        labels_a + labels_b, weights_a + [-w for w in weights_b]
    )
    assert np.allclose(
        result.to_sparse_matrix().toarray(), expected.to_sparse_matrix().toarray()
    )
    assert np.isclose(result.constant, expected.constant)


@pytest.mark.parametrize("random_pauli_hamiltonian", PARAMS, indirect=True)
def test_scalar_multiplication(random_pauli_hamiltonian):
    hamil, _, _ = random_pauli_hamiltonian
    result = 2.0 * hamil
    assert np.allclose(
        result.to_sparse_matrix().toarray(), 2.0 * hamil.to_sparse_matrix().toarray()
    )
    assert np.isclose(result.constant, 2.0 * hamil.constant)


@pytest.mark.parametrize("random_pauli_hamiltonian", PARAMS, indirect=True)
def test_spectrum(random_pauli_hamiltonian):
    hamil, _, _ = random_pauli_hamiltonian
    matrix = hamil._dense_matrix() + hamil.constant * np.eye(hamil.dimension)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    assert np.isclose(hamil.ground_state_energy(), eigvals[0])
    assert np.isclose(hamil.max_energy(), eigvals[-1])
    assert np.isclose(np.abs(np.vdot(hamil.ground_state(), eigvecs[:, 0])), 1.0)


def test_tensor_ordering():
    # "ZI" should correspond to Z⊗I, giving diagonal (1, 1, -1, -1)
    hamil = PauliHamiltonian.from_labels_and_weights(["ZI"], [1.0])
    diag = np.diag(hamil.to_sparse_matrix().toarray())
    assert np.allclose(
        diag, [1, 1, -1, -1]
    ), f"Got diagonal {diag}, expected (1, 1, -1, -1)"


def test_incompatible_add_raises():
    hamil_2q = PauliHamiltonian.from_labels_and_weights(["ZI", "IZ"], [1.0, -1.0])
    hamil_3q = PauliHamiltonian.from_labels_and_weights(["ZII", "IZI"], [1.0, -1.0])
    with pytest.raises(RuntimeError):
        _ = hamil_2q + hamil_3q
