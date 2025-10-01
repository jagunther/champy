from champy.Hamiltonian import Hamiltonian
from pyscf import fci
from pyscf.tools import fcidump
from pyscf.ao2mo import restore
import numpy as np
import scipy
from numba import jit


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

    def __eq__(self, other):
        if self._compatible(other):
            if np.allclose(other.h0, self.h0):
                if np.allclose(other.h1e, self.h1e):
                    if np.allclose(other.h2e, self.h2e):
                        return True
        return False

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

    @jit(nopython=True)
    def pauli_coeffs(self) -> (float, np.ndarray, np.ndarray):
        """
        Computes the coefficients of the Hamiltonian after Fermion-to-qubit mapping, i.e.
        for resulting qubit Hamiltonian Σ_i h_i P_i, where P_i are Paulistrings, it returns
        the coefficients h_i.

        Returns tuple (coeff_constant, coeffs quadr terms, coeffs quart terms)

        See equation F9 in https://quantum-journal.org/papers/q-2023-05-12-1000/
        """
        coeffs_quadr = []
        coeffs_quart = []

        coeff_const = self.h0
        for p in range(self.num_orb):
            coeff_const += self.h1e[p, p]
            for r in range(self.num_orb):
                coeff_const += 0.5 * self.h2e[p, p, r, r]
                coeff_const -= 0.25 * self.h2e[p, r, r, p]

        for p in range(self.num_orb):
            for q in range(self.num_orb):
                curr = self.h1e[p, q]
                for r in range(self.num_orb):
                    curr += self.h2e[p, q, r, r]
                    curr -= 0.5 * self.h2e[p, r, r, q]
                coeffs_quadr.append(1j / 2 * curr)
                coeffs_quadr.append(1j / 2 * curr)

        for r in range(self.num_orb - 1):
            for p in range(r + 1, self.num_orb):
                for q in range(self.num_orb - 1):
                    for s in range(q + 1, self.num_orb):
                        coeffs_quart.append(
                            (self.h2e[p, q, r, s] - self.h2e[p, s, r, q]) / 4
                        )
                        coeffs_quart.append(
                            (self.h2e[p, q, r, s] - self.h2e[p, s, r, q]) / 4
                        )

        for r in range(self.num_orb - 1):
            for p in range(r + 1, self.num_orb):
                for q in range(self.num_orb):
                    for s in range(q, self.num_orb):
                        coeffs_quart.append(self.h2e[p, q, r, s] / 4)
                        coeffs_quart.append(self.h2e[p, q, r, s] / 4)

        for r in range(self.num_orb):
            for p in range(r, self.num_orb):
                for q in range(1, self.num_orb):
                    for s in range(q):
                        coeffs_quart.append(self.h2e[p, q, r, s] / 4)
                        coeffs_quart.append(self.h2e[p, q, r, s] / 4)

        for p in range(self.num_orb):
            for q in range(self.num_orb):
                coeffs_quart.append(self.h2e[p, q, p, q] / 4)

        return coeff_const, np.array(coeffs_quadr), np.array(coeffs_quart)

    def sum_pauli_coeffs(self) -> float:
        """
        Compute the sum of absolute values of coefficients of Hamiltonian expressed in terms of Paulis
        after fermion-to-qubit mapping.
        """
        res = 0
        for p in range(self.num_orb):
            for q in range(self.num_orb):
                curr = self.h1e[p, q]
                for r in range(self.num_orb):
                    curr += self.h2e[p, q, r, r]
                    curr -= 0.5 * self.h2e[p, r, r, q]
                res += abs(curr)

        for r in range(self.num_orb - 1):
            for p in range(r + 1, self.num_orb):
                for q in range(self.num_orb - 1):
                    for s in range(q + 1, self.num_orb):
                        res += abs(self.h2e[p, q, r, s] - self.h2e[p, s, r, q]) / 2

        for r in range(self.num_orb - 1):
            for p in range(r + 1, self.num_orb):
                for q in range(self.num_orb):
                    for s in range(q, self.num_orb):
                        res += abs(self.h2e[p, q, r, s] / 2)

        for r in range(self.num_orb):
            for p in range(r, self.num_orb):
                for q in range(1, self.num_orb):
                    for s in range(q):
                        res += abs(self.h2e[p, q, r, s] / 2)

        for p in range(self.num_orb):
            for q in range(self.num_orb):
                res += abs(self.h2e[p, q, p, q] / 4)

        return res
