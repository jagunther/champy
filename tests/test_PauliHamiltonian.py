import numpy as np
import pytest
from pauliarray import Operator, WeightedPauliArray

from champy.PauliHamiltonian import PauliHamiltonian


def _sample_hamiltonian() -> PauliHamiltonian:
    wpaulis = WeightedPauliArray.from_labels_and_weights(
        ["II", "ZI", "IZ", "XX"], [0.5, 1.2, -0.7, 0.3]
    )
    return PauliHamiltonian(wpaulis)


def test_constant_and_dimension():
    hamil = _sample_hamiltonian()
    assert np.isclose(hamil.constant, 0.5)
    assert hamil.dimension == 4


def test_paulis_and_coeffs_attributes():
    hamil = _sample_hamiltonian()
    assert np.array_equal(hamil.paulis, np.array(["ZI", "IZ", "XX"]))
    assert np.allclose(hamil.coeffs, np.array([1.2, -0.7, 0.3]))


def test_to_sparse_matrix_excludes_constant():
    hamil = _sample_hamiltonian()
    expected = Operator(
        WeightedPauliArray.from_labels_and_weights(["ZI", "IZ", "XX"], [1.2, -0.7, 0.3])
    ).to_matrix()
    assert np.allclose(hamil.to_sparse_matrix().toarray(), expected)


def test_arithmetic_and_spectrum():
    hamil = _sample_hamiltonian()
    hamil_add = hamil + hamil
    hamil_sub = hamil - hamil
    hamil_mul = 2.0 * hamil

    assert np.allclose(hamil_add.to_sparse_matrix().toarray(), 2.0 * hamil.to_sparse_matrix().toarray())
    assert np.allclose(hamil_mul.to_sparse_matrix().toarray(), 2.0 * hamil.to_sparse_matrix().toarray())
    assert np.allclose(hamil_sub.to_sparse_matrix().toarray(), np.zeros((4, 4)))
    assert np.isclose(hamil_add.constant, 2.0 * hamil.constant)
    assert np.isclose(hamil_mul.constant, 2.0 * hamil.constant)
    assert np.isclose(hamil_sub.constant, 0.0)

    matrix = hamil.operator.to_matrix() + hamil.constant * np.eye(hamil.dimension)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    assert np.isclose(hamil.ground_state_energy(), eigvals[0])
    assert np.isclose(hamil.max_energy(), eigvals[-1])
    assert np.isclose(np.abs(np.vdot(hamil.ground_state(), eigvecs[:, 0])), 1.0)


def test_incompatible_add_raises():
    hamil_2q = _sample_hamiltonian()
    hamil_3q = PauliHamiltonian(
        WeightedPauliArray.from_labels_and_weights(["III", "ZII"], [1.0, -2.0])
    )
    with pytest.raises(RuntimeError):
        _ = hamil_2q + hamil_3q
