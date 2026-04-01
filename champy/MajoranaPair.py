from champy.Hamiltonian import Hamiltonian
from champy.PauliHamiltonian import PauliHamiltonian
import numpy as np
import matplotlib.pyplot as plt


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
            - 0.25 * np.einsum("prrp->", h2e)
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
        self.f2e_sameop_diffspin = np.where((p == r) & (q == s), h2e, 0) / 4
        self.f2e_sameop_diffspin = np.einsum("pqpq->pq", self.f2e_sameop_diffspin)

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

    def commutation_graph(self):
        """
        Compute adjacency matrix of all 2n^2 Majorana Pairs.
        Operators are ordered as (p,q,σ) with flat index σ*n²+p*n+q,
        so spin-up block comes first, spin-down second.

        Two pairs (pq,σ) and (rs,τ) anticommute iff σ==τ and (p==r or q==s).
        """
        n = self.num_orb
        p, q, r, s = np.ogrid[:n, :n, :n, :n]
        # same-spin anticommutation block: shape (n², n²)
        same_spin_block = ((p == r) | (q == s)).reshape(n * n, n * n).astype(np.int8)

        adj = np.zeros((2 * n * n, 2 * n * n), dtype=np.int8)
        adj[: n * n, : n * n] = same_spin_block  # spin-up vs spin-up
        adj[n * n :, n * n :] = same_spin_block  # spin-down vs spin-down
        return adj

    def majoranapair_index(self, p: int, q: int, sigma: int) -> int:
        """Return the flat index of Γ_{pq,σ} in the commutation graph adjacency matrix.

        Index = sigma * n² + p * n + q,  sigma ∈ {0, 1}.
        """
        n = self.num_orb
        return sigma * n * n + p * n + q

    def majoranapair_weights(self) -> np.ndarray:

        n = self.num_orb
        weights = np.zeros((n, n, 2))
        weights += np.abs(self.f1e)[:, :, np.newaxis]

        for f in [self.f2e_diffopp_samespin, self.f2e_diffop_diffspin]:
            weights += np.einsum("pqrs->pq", np.abs(f))[:, :, np.newaxis]
            weights += np.einsum("pqrs->rs", np.abs(f))[:, :, np.newaxis]

        weights += np.abs(self.f2e_sameop_diffspin)[:, :, np.newaxis]
        return weights

    def plot_orbital_graph(self) -> None:
        """Plot the orbital graph for the spin-↑ sector.

        Γ_pp → vertex p  (color proportional to weight)
        Γ_pq → undirected edge  (color proportional to weight of (p,q))
        """
        import matplotlib.colors as mcolors
        import matplotlib.colorbar as mcolorbar

        n = self.num_orb
        w = self.majoranapair_weights()[:, :, 0]

        cmap = plt.colormaps["turbo"]
        diag_vals = w[np.arange(n), np.arange(n)]
        edge_vals = np.array([w[p, q] for p in range(n) for q in range(p + 1, n)])
        norm = mcolors.Normalize(vmin=0, vmax=w.max())

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(right=0.78)

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
        pos = {p: np.array([np.cos(angles[p]), np.sin(angles[p])]) for p in range(n)}

        # Undirected edges (p < q to draw each pair once)
        for p in range(n):
            for q in range(p + 1, n):
                val = w[p, q]
                if val < 1e-6 * edge_vals.max():
                    continue
                xs = [pos[p][0], pos[q][0]]
                ys = [pos[p][1], pos[q][1]]
                ax.plot(xs, ys, color=cmap(norm(val)), lw=2, zorder=1)

        # Vertices
        NODE_RADIUS = 0.12
        for p in range(n):
            circle = plt.Circle(
                pos[p],
                NODE_RADIUS,
                facecolor=cmap(norm(diag_vals[p])),
                edgecolor="black",
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(circle)
            r, g, b, _ = cmap(norm(diag_vals[p]))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.5 else "black"
            ax.text(
                *pos[p],
                str(p),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=text_color,
                zorder=4,
            )

        # Colorbar
        cax = fig.add_axes([0.82, 0.15, 0.03, 0.7])
        mcolorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
        cax.set_title(r"$w$", fontsize=10)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            "Orbital graph (spin-↑ sector)\n"
            r"$\Gamma_{pp}$ → vertex,  $\Gamma_{pq}$ → edge",
            fontsize=11,
        )
        plt.show()

    def jordan_wigner(self) -> PauliHamiltonian:
        n = self.num_orb
        total_qubits = 2 * n
        labels = []
        coeffs = []

        # constant
        labels.append("I" * total_qubits)
        coeffs.append(self.constant)

        # 1-electron: f1e[p,q] maps to one Pauli per spin sector
        for p in range(n):
            for q in range(n):
                if self.f1e[p, q] == 0:
                    continue
                for spin_offset in (0, n):
                    phase, l = _jw_pauli(p, q, spin_offset, total_qubits)
                    labels.append(l)
                    coeffs.append(phase * self.f1e[p, q])

        # 2-electron same spin, different operators: Γ_{pq,σ} Γ_{rs,σ}
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        coeff = self.f2e_diffopp_samespin[p, q, r, s]
                        if coeff == 0:
                            continue
                        for spin_offset in (0, n):
                            phase1, l1 = _jw_pauli(p, q, spin_offset, total_qubits)
                            phase2, l2 = _jw_pauli(r, s, spin_offset, total_qubits)
                            phase, label = _multiply_pauli_strings(l1, l2)
                            labels.append(label)
                            coeffs.append(coeff * phase1 * phase2 * phase)

        # 2-electron different spin, different operators: Γ_{pq,σ} Γ_{rs,τ}
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        coeff = self.f2e_diffop_diffspin[p, q, r, s]
                        if coeff == 0:
                            continue
                        for spin_offset in [(0, n), (n, 0)]:
                            phase1, l1 = _jw_pauli(p, q, spin_offset[0], total_qubits)
                            phase2, l2 = _jw_pauli(r, s, spin_offset[1], total_qubits)
                            phase, label = _multiply_pauli_strings(l1, l2)
                            labels.append(label)
                            coeffs.append(coeff * phase1 * phase2 * phase)

        # 2-electron same operator, different spin: Γ_{pq,↑} Γ_{pq,↓}
        for p in range(n):
            for q in range(n):
                coeff = self.f2e_sameop_diffspin[p, q]
                if coeff == 0:
                    continue
                phase1, l1 = _jw_pauli(p, q, 0, total_qubits)
                phase2, l2 = _jw_pauli(p, q, n, total_qubits)
                phase, label = _multiply_pauli_strings(l1, l2)
                labels.append(label)
                coeffs.append(coeff * phase1 * phase2 * phase)

        return PauliHamiltonian.from_labels_and_weights(labels, coeffs)


def _jw_pauli(p, q, spin_offset, total_qubits):
    """Pauli string for Majorana pair operator Γ_{pq,σ} in a given spin sector
        in the Jordan-Wigner encoding.

    p == q: Z_p
    p < q:  Y_p Z_{p+1}...Z_{q-1} Y_q
    p > q:  X_q Z_{q+1}...Z_{p-1} X_p
    """
    chars = ["I"] * total_qubits
    phase = None
    if p == q:
        chars[spin_offset + p] = "Z"
        phase = -1
    else:
        phase = 1
        lo, hi = min(p, q), max(p, q)
        for k in range(lo + 1, hi):
            chars[spin_offset + k] = "Z"
    if p > q:
        chars[spin_offset + p] = "X"
        chars[spin_offset + q] = "X"
    if p < q:
        chars[spin_offset + p] = "Y"
        chars[spin_offset + q] = "Y"
    return phase, "".join(chars)


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
