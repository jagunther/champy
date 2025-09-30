from abc import ABC, abstractmethod
import scipy


class Hamiltonian(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __sub__(self, other):
        pass

    @abstractmethod
    def __mul__(self, other):
        pass  # multiplication by scalar

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def to_sparse_matrix(self) -> scipy.sparse.csr_matrix:
        pass

    @property
    @abstractmethod
    def constant(self) -> float:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def ground_state_energy(self) -> float:
        pass

    @abstractmethod
    def max_energy(self) -> float:
        pass

    def spectral_range(self) -> float:
        return self.max_energy() - self.ground_state_energy()

    @abstractmethod
    def ground_state(self) -> scipy.sparse.csr_matrix:
        pass
