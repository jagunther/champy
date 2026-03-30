from numbers import Number

import numpy as np
import scipy
from champy.Hamiltonian import Hamiltonian
from pauliarray import Operator
from pauliarray.binary import bit_operations as bitops
from pauliarray.pauli import pauli_array as pa


class PauliHamiltonian(Hamiltonian):

    def __init__(self, operator: Operator):
        if not isinstance(operator, Operator):
            raise TypeError("PauliHamiltonian must be initialized with an Operator.")
        wpaulis = operator.wpaulis
        is_identity = wpaulis.paulis.is_identity()
        self._constant = float(np.real_if_close(np.sum(wpaulis.weights[is_identity])))
        self._operator = Operator(wpaulis.extract(~is_identity))
        super().__init__()

    @classmethod
    def from_labels_and_weights(cls, labels, weights) -> "PauliHamiltonian":
        return cls(Operator.from_labels_and_weights(labels, weights))

    def _compatible(self, other) -> bool:
        return isinstance(other, PauliHamiltonian) and (
            self._operator.wpaulis.num_qubits == other._operator.wpaulis.num_qubits
        )

    @property
    def paulis(self) -> np.ndarray:
        return self._operator.wpaulis.paulis.to_labels().copy()

    @property
    def coeffs(self) -> np.ndarray:
        return self._operator.wpaulis.weights.copy()

    def __add__(self, other):
        if not self._compatible(other):
            raise RuntimeError(
                "PauliHamiltonian objects must have same number of qubits when adding!"
            )
        result = PauliHamiltonian((self._operator + other._operator).remove_small_weights())
        result._constant = self._constant + other._constant
        return result

    def __sub__(self, other):
        if not self._compatible(other):
            raise RuntimeError(
                "PauliHamiltonian objects must have same number of qubits when subtracting!"
            )
        result = PauliHamiltonian((self._operator + other._operator * -1).remove_small_weights())
        result._constant = self._constant - other._constant
        return result

    def __mul__(self, other):
        if not isinstance(other, Number):
            raise TypeError(f"Cannot multiply PauliHamiltonian by {type(other)}!")
        result = PauliHamiltonian((self._operator * other).remove_small_weights())
        result._constant = self._constant * other
        return result

    __rmul__ = __mul__

    def __eq__(self, other) -> bool:
        if not self._compatible(other):
            return False
        return (
            np.isclose(self._constant, other._constant)
            and np.allclose(self.to_sparse_matrix().toarray(), other.to_sparse_matrix().toarray())
        )

    @property
    def constant(self) -> float:
        return self._constant

    @property
    def dimension(self) -> int:
        return 2 ** self._operator.wpaulis.num_qubits

    def to_sparse_matrix(self) -> scipy.sparse.csr_matrix:
        """Returns the matrix representation of the Hamiltonian, excluding the constant."""
        wpaulis = self._operator.wpaulis
        num_qubits = wpaulis.num_qubits
        dim = 2 ** num_qubits
        n_terms = wpaulis.shape[0]

        z_ints = bitops.strings_to_ints(wpaulis.paulis.z_strings)
        x_ints = bitops.strings_to_ints(wpaulis.paulis.x_strings)
        phase_powers = np.mod(bitops.dot(wpaulis.paulis.z_strings, wpaulis.paulis.x_strings), 4)
        phases = np.choose(phase_powers, [1, -1j, -1, 1j])

        rows = np.empty(n_terms * dim, dtype=np.int64)
        cols = np.empty(n_terms * dim, dtype=np.int64)
        vals = np.empty(n_terms * dim, dtype=complex)

        for i in range(n_terms):
            row_ind, col_ind, matrix_elements = pa.PauliArray.sparse_matrix_from_zx_ints(
                z_ints[i], x_ints[i], num_qubits
            )
            s = slice(i * dim, (i + 1) * dim)
            rows[s] = row_ind
            cols[s] = col_ind
            vals[s] = wpaulis.weights[i] * phases[i] * matrix_elements

        return scipy.sparse.csr_matrix(
            scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(dim, dim))
        )

    def _dense_matrix(self) -> np.ndarray:
        """Dense matrix, intended for testing the sparse implementation."""
        return self._operator.to_matrix()

    def ground_state_energy(self) -> float:
        eigval, _ = scipy.sparse.linalg.eigsh(self.to_sparse_matrix(), k=1, which="SA")
        return float(np.real_if_close(eigval[0])) + self.constant

    def max_energy(self) -> float:
        eigval, _ = scipy.sparse.linalg.eigsh(self.to_sparse_matrix(), k=1, which="LA")
        return float(np.real_if_close(eigval[0])) + self.constant

    def ground_state(self) -> np.ndarray:
        _, eigvec = scipy.sparse.linalg.eigsh(self.to_sparse_matrix(), k=1, which="SA")
        return eigvec[:, 0]
