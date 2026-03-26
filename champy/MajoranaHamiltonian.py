from champy.Hamiltonian import Hamiltonian
import numpy as np


class MajoranaHamiltonian(Hamiltonian):

    def __init__(
        self,
        const: float,
        coeff2: np.ndarray,
        coeff4: np.ndarray,
    ):

        assert coeff2.shape == 2
        assert coeff4.shape == 4
        assert coeff2.shape[0] % 2 == 0
        assert set(coeff2.shape) == set(coeff4.shape)

        self._constant = const

    def _compatible(self, other):
        assert self.num_qubits == other.num_qubits

    def __add__(self, other):
        if self._compatible(other):
            return MajoranaHamiltonian(
                const=self.constant + other.constant,
                coeff2=self.coeff2 + other.coeff2,
                coeff4=self.coeff4 + other.coeff4,
            )
        else:
            raise RuntimeError(
                "Majorana Hamiltonians must have same number of qubits when adding!"
            )

    def __sub__(self, other):
        if self._compatible(other):
            return MajoranaHamiltonian(
                const=self.constant - other.constant,
                coeff2=self.coeff2 - other.coeff2,
                coeff4=self.coeff4 - other.coeff4,
            )
        else:
            raise RuntimeError(
                "Majorana Hamiltonians must have same number of qubits when subtracting!"
            )

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return MajoranaHamiltonian(
                const=self.constant * other,
                ind2=self.ind2,
                ind4=self.ind4,
                coeff2=self.coeff2 * other,
                coeff4=self.coeff4 * other,
            )
        else:
            raise TypeError(f"Cannot multiply MajoranaHamiltonian by {type(other)}!")

    def __eq__(self, other):
        if self._compatible(other):
            if np.allclose(self.constant, other.constant):
                if np.allclose(self.coeff2, other.coeff2):
                    if np.allclose(self.coeff4, other.coeff4):
                        return True
        else:
            return False

    @property
    def constant(self) -> float:
        return self._constant

    def to_sparse_matrix(self):
        raise NotImplementedError

    @property
    def num_qubits(self) -> int:
        return self.coeff2.shape[0]

    @property
    def dimension(self) -> int:
        return int(2**self.num_qubits)

    def ground_state_energy(self) -> float:
        raise NotImplementedError

    def max_energy(self) -> float:
        raise NotImplementedError

    def ground_state(self):
        raise NotImplementedError
