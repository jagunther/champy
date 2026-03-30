from champy.Hamiltonian import Hamiltonian
from champy.PauliHamiltonian import PauliHamiltonian
import numpy as np


class MajoranaPair(Hamiltonian):

    def __init__(
        self,
        h0: float,
        h1e: np.ndarray,
        h2e: np.ndarray,
    ):

        assert len(h1e.shape) == 2
        assert len(h2e.shape) == 4
        assert set(h1e.shape) == set(h2e.shape)

        self.h1e = h1e
        self.h2e = h2e

        # constant: h0 + trace(h1e) + one-body part from h2e
        self._constant = float(
            h0
            + np.trace(h1e)
            + 0.5 * np.einsum("pprr->", h2e)
            - 0.5 * np.einsum("prrp->", h2e)
        )

        # 1-el Majorana coefficients: Γ_pq,σ
        self.f1e = (
            h1e
            + np.einsum("pqrr->pq", self.h2e)
            - 0.5 * np.einsum("prrq->pq", self.h2e)
        ) / 2

        # 2-el Majorana coefficients, diff operators, same spin: Γ_pq,σ Γ_rs,σ
        n = self.num_orb
        p, q, r, s = np.ogrid[:n, :n, :n, :n]
        self.f2e_diffopp_samespin = (
            np.where((p > r) & (q < s), h2e - np.swapaxes(h2e, 1, 3), 0) / 4
        )

        # 2-el Majorana coefficients, diff operators, diff spin: Γ_pq,σ Γ_rs,τ
        self.f2e_diffop_diffspin = np.where((p > r) & (q <= s), h2e, 0) / 4
        self.f2e_diffop_diffspin += np.where((p >= r) & (q > s), h2e, 0) / 4

        # 2-el Majorana coefficients, same operators, diff spin: Γ_pq,σ Γ_pq,τ
        self.f2e_sameop_diffspin = np.where((p == r) & (q == s), h2e, 0)

    def _compatible(self, other):
        assert self.num_orb == other.num_orb

    def __add__(self, other):
        if self._compatible(other):
            return MajoranaPair(
                h0=0, h1e=self.h1e + other.h1e, h2e=self.h2e + other.h2e
            )
        else:
            raise RuntimeError(
                "Majorana Hamiltonians must have same number of qubits when adding!"
            )

    def __sub__(self, other):
        if self._compatible(other):
            return MajoranaPair(
                h0=0, h1e=self.h1e - other.h1e, h2e=self.h2e - other.h2e
            )
        else:
            raise RuntimeError(
                "Majorana Hamiltonians must have same number of qubits when subtracting!"
            )

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return MajoranaPair(h0=0, h1e=self.h1e * other, h2e=self.h2e * other)
        else:
            raise TypeError(f"Cannot multiply MajoranaPair by {type(other)}!")

    def __eq__(self, other):
        if self._compatible(other):
            if np.allclose(self.constant, other.constant):
                if np.allclose(self.h1e, other.coeff2):
                    if np.allclose(self.h2e, other.coeff4):
                        return True
        else:
            return False

    @property
    def constant(self) -> float:
        return self._constant

    def to_sparse_matrix(self):
        raise NotImplementedError

    @property
    def num_orb(self) -> int:
        return self.h1e.shape[0]

    @property
    def dimension(self) -> int:
        return int(2**self.num_orb)

    def ground_state_energy(self) -> float:
        raise NotImplementedError

    def max_energy(self) -> float:
        raise NotImplementedError

    def ground_state(self):
        raise NotImplementedError

    def jordan_wigner(self) -> PauliHamiltonian:
        n = self.num_orb
        total_qubits = 2 * n
        labels = []
        weights = []

        # constant
        labels.append("I" * total_qubits)
        weights.append(self.constant)

        # 1-electron: f1e[p,q] maps to one Pauli per spin sector
        for p in range(n):
            for q in range(n):
                if self.f1e[p, q] == 0:
                    continue
                for spin_offset in (0, n):
                    labels.append(_jw_pauli_label(p, q, spin_offset, total_qubits))
                    if p == q:
                        weights.append(self.f1e[p, q])
                    else:
                        weights.append(-self.f1e[p, q])

        # 2-electron same spin, different operators: Γ_{pq,σ} Γ_{rs,σ}
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        coeff = self.f2e_diffopp_samespin[p, q, r, s]
                        if coeff == 0:
                            continue
                        for spin_offset in (0, n):
                            l1 = _jw_pauli_label(p, q, spin_offset, total_qubits)
                            l2 = _jw_pauli_label(r, s, spin_offset, total_qubits)
                            phase, label = _multiply_pauli_strings(l1, l2)
                            labels.append(label)
                            weights.append(coeff * phase)

        # 2-electron different spin, different operators: Γ_{pq,↑} Γ_{rs,↓}
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        coeff = self.f2e_diffop_diffspin[p, q, r, s]
                        if coeff == 0:
                            continue
                        l_up = _jw_pauli_label(p, q, 0, total_qubits)
                        l_down = _jw_pauli_label(r, s, n, total_qubits)
                        phase, label = _multiply_pauli_strings(l_up, l_down)
                        labels.append(label)
                        weights.append(coeff * phase)

        # 2-electron same operator, different spin: Γ_{pq,↑} Γ_{pq,↓}
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        coeff = self.f2e_sameop_diffspin[p, q, r, s]
                        if coeff == 0:
                            continue
                        l_up = _jw_pauli_label(p, q, 0, total_qubits)
                        l_down = _jw_pauli_label(r, s, n, total_qubits)
                        phase, label = _multiply_pauli_strings(l_up, l_down)
                        labels.append(label)
                        weights.append(coeff * phase)

        print(labels)
        return PauliHamiltonian.from_labels_and_weights(labels, weights)


def _jw_pauli_label(p, q, spin_offset, total_qubits):
    """Pauli string for Majorana pair operator Γ_{pq,σ} in a given spin sector
        in the Jordan-Wigner encoding.

    p == q: Z_p
    p < q:  Y_p Z_{p+1}...Z_{q-1} Y_q
    p > q:  X_q Z_{q+1}...Z_{p-1} X_p
    """
    chars = ["I"] * total_qubits
    if p == q:
        chars[spin_offset + p] = "Z"
    else:
        lo, hi = min(p, q), max(p, q)
        for k in range(lo + 1, hi):
            chars[spin_offset + k] = "Z"
    if p > q:
        chars[spin_offset + p] = "X"
        chars[spin_offset + q] = "X"
    if p < q:
        chars[spin_offset + p] = "Y"
        chars[spin_offset + q] = "Y"
    return "".join(chars)


_PAULI_PRODUCT = {
    ("X", "Y"): (1j, "Z"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "X"): (1j, "Y"),
    ("X", "Z"): (-1j, "Y"),
}


def _multiply_pauli_strings(s1, s2):
    """Multiply two same-length Pauli strings. Returns (phase, label)."""
    phase = 1.0 + 0j
    chars = []
    for a, b in zip(s1, s2):
        if a == "I":
            chars.append(b)
        elif b == "I":
            chars.append(a)
        elif a == b:
            chars.append("I")
        else:
            p, c = _PAULI_PRODUCT[(a, b)]
            phase *= p
            chars.append(c)
    return phase, "".join(chars)
