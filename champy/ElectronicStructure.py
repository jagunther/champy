from champy.Hamiltonian import Hamiltonian
from pyscf import fci
from pyscf.tools import fcidump
from pyscf.ao2mo import restore
import numpy as np
import scipy


class ElectronicStructure(Hamiltonian):

    def __init__(self, h0: float, h1e: np.ndarray, h2e, num_elec: int):

        # assert h1e and h2e have consistent shapes
        assert len(h1e.shape) == 2
        assert len(h2e.shape) == 4
        assert len(list(set(h1e.shape))) == 1
        assert set(h1e.shape) == set(h2e.shape)

        self.h0 = h0
        self.h1e = h1e
        self.h2e = h2e
        self.num_elec = num_elec
        super().__init__()

    def _compatible(self, other):
        if self.num_elec == other.num_elec and self.num_orb == other.num_orb:
            return True
        else:
            return False

    def __add__(self, other):
        if self._compatible(other):
            return ElectronicStructure(
                self.h0 + other.h0,
                self.h1e + other.h1e,
                self.h2e + other.h2e,
                self.num_elec,
            )
        else:
            raise RuntimeError(
                "ElectronicStructure objects must have same number of electrons when adding!"
            )

    def __sub__(self, other):
        if self._compatible(other):
            return ElectronicStructure(
                self.h0 - other.h0,
                self.h1e - other.h1e,
                self.h2e - other.h2e,
                self.num_elec,
            )
        else:
            raise RuntimeError(
                "ElectronicStructure objects must have same number of orbitals when adding!"
            )

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return ElectronicStructure(
                other * self.h0, other * self.h1e, other * self.h2e, self.num_elec
            )
        else:
            raise TypeError(f"Cannot multiply ElectronicStructure by {type(other)}!")

    __rmul__ = __mul__

    @property
    def constant(self) -> float:
        return self.h0

    @property
    def num_orb(self) -> int:
        return self.h1e.shape[0]

    @property
    def dimension(self) -> int:
        return (2 * self.num_orb) ** 2

    def to_sparse_matrix(self) -> scipy.sparse.csr_matrix:
        """
        Does not contain constant
        """
        return fci.direct_spin1.pspace(
            self.h1e, self.h2e, self.num_orb, self.num_elec, np=self.dimension
        )[1]

    def ground_state_energy(self) -> float:
        # The spin1 method works generally, see github.com/pyscf/pyscf/blob/master/examples/fci/01-given_h1e_h2e.py
        cisolver = fci.direct_spin1.FCI()
        cisolver.max_cycle = 100
        cisolver.conv_tol = 1e-10
        e, _ = cisolver.kernel(self.h1e, self.h2e, self.num_orb, self.num_elec)
        return e + self.constant

    def max_energy(self) -> float:
        cisolver = fci.direct_spin1.FCI()
        cisolver.max_cycle = 100
        cisolver.conv_tol = 1e-10
        e, _ = cisolver.kernel(-self.h1e, -self.h2e, self.num_orb, self.num_elec)
        return e - self.constant

    def ground_state(self) -> scipy.sparse.csr_matrix:
        _, fcivec = fci.direct_spin1.kernel(
            self.h1e,
            self.h2e,
            self.num_orb,
            self.num_elec,
            nroots=3,
            max_space=50,
            max_cycle=200,
        )
        return fcivec[0].flatten()

    @staticmethod
    def from_fcidump(file: str):
        """
        Create an ElectronicStructure object from an FCIDUMP file
        :param file: path to the FCIDUMP file
        :return: ElectronicStructure object
        """
        data = fcidump.read(file)
        h0 = data["ECORE"]
        h1e = data["H1"]
        h2e = data["H2"]
        norb = data["NORB"]
        nelec = data["NELEC"]
        h2e = restore("s1", h2e, norb)
        return ElectronicStructure(h0=h0, h1e=h1e, h2e=h2e, num_elec=nelec)

    def to_fcidump(self, file: str):
        """
        Write the ElectronicStructure object to an FCIDUMP file

        :param file: path to file
        """
        fcidump.from_integrals(
            filename=file,
            h1e=self.h1e,
            h2e=self.h2e,
            nmo=self.num_orb,
            nelec=self.num_elec,
            nuc=self.constant,
        )
