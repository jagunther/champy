from numbers import Number

import numpy as np
import scipy
from champy.Hamiltonian import Hamiltonian
from pauliarray import Operator, WeightedPauliArray


class PauliHamiltonian(Hamiltonian):

    def __init__(self, weighted_paulis: WeightedPauliArray):
        if not isinstance(weighted_paulis, WeightedPauliArray):
            raise TypeError(
                "PauliHamiltonian must be initialized with a WeightedPauliArray."
            )
        # A Hamiltonian is a sum of Pauli terms, so we keep a 1D representation.
        flattened = weighted_paulis.flatten().copy()
        is_identity = flattened.paulis.is_identity()
        constant = np.sum(flattened.weights[is_identity])
        self._constant = float(np.real_if_close(constant))
        self.weighted_paulis = flattened[~is_identity].copy()
        super().__init__()

    def _compatible(self, other) -> bool:
        return isinstance(other, PauliHamiltonian) and (
            self.weighted_paulis.num_qubits == other.weighted_paulis.num_qubits
        )

    @property
    def operator(self) -> Operator:
        return Operator(self.weighted_paulis)

    @property
    def paulis(self) -> np.ndarray:
        return self.weighted_paulis.paulis.to_labels().copy()

    @property
    def coeffs(self) -> np.ndarray:
        return self.weighted_paulis.weights.copy()

    def __add__(self, other):
        if not self._compatible(other):
            raise RuntimeError(
                "PauliHamiltonian objects must have same number of qubits when adding!"
            )
        result = PauliHamiltonian((self.operator + other.operator).wpaulis)
        result._constant = self.constant + other.constant
        return result

    def __sub__(self, other):
        if not self._compatible(other):
            raise RuntimeError(
                "PauliHamiltonian objects must have same number of qubits when subtracting!"
            )
        result = PauliHamiltonian((self.operator - other.operator).wpaulis)
        result._constant = self.constant - other.constant
        return result

    def __mul__(self, other):
        if not isinstance(other, Number):
            raise TypeError(f"Cannot multiply PauliHamiltonian by {type(other)}!")
        result = PauliHamiltonian((self.operator * other).wpaulis)
        result._constant = self.constant * other
        return result

    __rmul__ = __mul__

    @property
    def constant(self) -> float:
        return self._constant

    @property
    def dimension(self) -> int:
        return 2 ** self.weighted_paulis.num_qubits

    def _matrix(self) -> np.ndarray:
        return self.operator.to_matrix()

    def to_sparse_matrix(self) -> scipy.sparse.csr_matrix:
        """
        Does not contain constant.
        """
        return scipy.sparse.csr_matrix(self._matrix())

    def ground_state_energy(self) -> float:
        eigvals = np.linalg.eigvalsh(self._matrix()) + self.constant
        return float(np.real_if_close(np.min(eigvals)))

    def max_energy(self) -> float:
        eigvals = np.linalg.eigvalsh(self._matrix()) + self.constant
        return float(np.real_if_close(np.max(eigvals)))

    def ground_state(self) -> scipy.sparse.csr_matrix:
        _, eigvecs = np.linalg.eigh(self._matrix())
        return eigvecs[:, 0]
