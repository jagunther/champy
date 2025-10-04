from champy.Hamiltonian import Hamiltonian
from pyscf import fci
from pyscf.tools import fcidump
from pyscf.ao2mo import restore
from pyscf.mcscf import CASCI
import numpy as np
import scipy
from numba import jit
import copy


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

    def fock_operator(self) -> np.ndarray:
        """
        Construct the Fock operator from the h1e and h2e

        :return: fock operator, num_orb x num_orb np.array
        """
        assert self.num_elec % 2 == 0
        num_occ = self.num_elec // 2
        fock_op = copy.deepcopy(self.h1e)
        fock_op += 2 * np.einsum("ijkk -> ij", self.h2e[:, :, :num_occ, :num_occ])
        fock_op -= np.einsum("ikkj -> ij", self.h2e[:, :num_occ, :num_occ, :])
        return fock_op

    def hf_orbital_energies(self) -> np.ndarray:
        """
        If h1e and h2e are given in the canonical HF basis, returns the orbital energies.
        """
        if not self.is_canonical_hf_basis():
            raise RuntimeError("h1e, h2e not in canonical HF basis")
        return np.diag(self.fock_operator())

    def is_canonical_hf_basis(self) -> bool:
        """
        Checks whether the underlying basis of h1e, h2e is the canonical HF basis
        """
        f = self.fock_operator()
        np.fill_diagonal(f, 0)
        if not np.all(np.abs(f) < 1e-6):
            return False
        if not np.allclose(np.diag(f), sorted(np.diag(f))):
            print("The HF orbitals are not in the right order")
            return False
        return True

    def hf_energy(self) -> float:
        """
        Compute the HF energy of the system. Checks first whether underlying orbitals are canonical
        HF orbitals. Assumes that orbitals are ordered by orbital energy.

        :return: HF energy
        """
        assert self.num_elec % 2 == 0
        num_occ = self.num_elec // 2
        if not self.is_canonical_hf_basis():
            raise RuntimeError(f"Not in canonical orbital basis")
        else:
            e = self.constant
            e += 2 * np.sum(np.diag(self.h1e)[:num_occ])
            for i in range(num_occ):
                for j in range(num_occ):
                    e += 2 * self.h2e[i, i, j, j]
                    e -= self.h2e[i, j, j, i]
            return e

    def hf_state(self) -> np.ndarray:
        """
        Computes the HF state of the system in the Fock basis (as groundstate)
        """
        # following https://github.com/pyscf/pyscf/issues/2154
        if not self.is_canonical_hf_basis():
            raise RuntimeError(f"Not in canonical orbital basis")
        assert self.num_elec % 2 == 0
        num_det_alpha = int(scipy.special.binom(self.num_orb, self.num_elec // 2))
        num_det_beta = int(scipy.special.binom(self.num_orb, self.num_elec // 2))
        hf_det = np.zeros((num_det_alpha, num_det_beta))
        hf_det[0, 0] = 1
        return hf_det.flatten()

    def hf_overlap(self) -> float:
        """
        Computes overlap between the HF state and the ground state
        """
        return np.abs(self.hf_state().T @ self.ground_state()) ** 2

    @staticmethod
    def from_pyscf(rhf, num_orb, num_elec):
        """
        Create an ElectronicStructure object from a PySCF RHF object

        :param rhf: converged PySCF RHF object
        :param num_orb: int, number of active spatial orbitals
        :param num_elec: int, number of active electrons
        :return: ElectronicStructure object
        """
        casci = CASCI(rhf, num_orb, num_elec)
        h1e, h0 = casci.get_h1eff()
        h2e = restore("s1", casci.get_h2eff(), num_orb)
        return ElectronicStructure(h0=h0, h1e=h1e, h2e=h2e, num_elec=num_elec)

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
