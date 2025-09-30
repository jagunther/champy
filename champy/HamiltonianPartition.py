from champy.Hamiltonian import Hamiltonian
from abc import ABCMeta, abstractmethod


class HamiltonianPartition(Hamiltonian, metaclass=ABCMeta):

    def __init__(self, hamiltonian_trunc: Hamiltonian, hamiltonian_rem: Hamiltonian):

        assert hamiltonian_trunc.dimension == hamiltonian_rem.dimension
        self.hamiltonian_trunc = hamiltonian_trunc
        self.hamiltonian_rem = hamiltonian_rem
        super().__init__()

    @property
    @abstractmethod
    def hamiltonian_full(self) -> Hamiltonian:
        pass
