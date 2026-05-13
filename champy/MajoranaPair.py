from champy.Hamiltonian import Hamiltonian
from champy.PauliHamiltonian import PauliHamiltonian
import functools
import numpy as np
import scipy.sparse


@functools.lru_cache(maxsize=16)
def _majorana_masks(n: int):
    """Boolean index masks for _majorana_coeffs, cached by n.

    Both masks depend only on the system size n (never on h1e/h2e), so they
    are computed once per unique n and reused across calls — or baked in as
    compile-time constants when the function is run under jax.jit.
    """
    idx = np.arange(n)
    p = idx[:, None, None, None]
    q = idx[None, :, None, None]
    r = idx[None, None, :, None]
    s = idx[None, None, None, :]
    # (p>r)&(q<s): diff operators, same spin
    mask_ss = (p > r) & (q < s)
    # (p>r)|(p==r and q>s): diff operators, diff spin (simplified from two conditions)
    mask_ds = (p > r) | ((p == r) & (q > s))
    return mask_ss, mask_ds


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

        (
            self._constant,
            self.f1e,
            self.f2e_diffopp_samespin,
            self.f2e_diffop_diffspin,
            self.f2e_sameop_diffspin,
        ) = MajoranaPair._majorana_coeffs(h0, h1e, h2e)

        # lazy cache for jw_matrix()
        self._jw_M = None
        self._jw_pauli_labels = None
        self._jw_col_labels = None

    @staticmethod
    @functools.partial(__import__("jax").jit, static_argnums=())
    def _majorana_coeffs(h0: float, h1e, h2e) -> tuple:
        """Compute the constant and Majorana coefficient tensors from h0, h1e, h2e.

        JIT-compiled and JAX-differentiable. Boolean masks are cached by n via
        _majorana_masks() and treated as compile-time constants under jax.jit.

        Returns
        -------
        constant : scalar
        f1e : array, shape (n, n)
        f2e_diffopp_samespin : array, shape (n, n, n, n)
        f2e_diffop_diffspin : array, shape (n, n, n, n)
        f2e_sameop_diffspin : array, shape (n, n)
        """
        import jax.numpy as jnp

        n = h1e.shape[0]
        mask_ss, mask_ds = _majorana_masks(n)

        constant = (
            h0
            + jnp.trace(h1e)
            + 0.5 * jnp.einsum("pprr->", h2e)
            - 0.25 * jnp.einsum("prrp->", h2e)
        )

        f1e = (
            h1e + jnp.einsum("pqrr->pq", h2e) - 0.5 * jnp.einsum("prrq->pq", h2e)
        ) / 2
        f2e_diffopp_samespin = jnp.where(mask_ss, h2e - jnp.swapaxes(h2e, 1, 3), 0) / 4
        f2e_diffop_diffspin = jnp.where(mask_ds, h2e, 0) / 4

        # einsum("pqpq->pq", A) picks A[p,q,p,q] without summation — the where is redundant
        f2e_sameop_diffspin = jnp.einsum("pqpq->pq", h2e) / 4

        return (
            constant,
            f1e,
            f2e_diffopp_samespin,
            f2e_diffop_diffspin,
            f2e_sameop_diffspin,
        )

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

    def jw_cost(self, perm: np.ndarray) -> float:
        """Cost of a JW ordering given as a permutation of orbital indices.

        cost = Σ_{p<q} w[p,q] * |pos[p] - pos[q]|

        where pos[p] is the position of orbital p in the JW string.
        """
        w = self.majoranapair_weights()[:, :, 0]
        n = len(perm)
        pos = np.empty(n, dtype=int)
        pos[perm] = np.arange(n)
        p_idx, q_idx = np.triu_indices(n, k=1)
        return float(np.sum(w[p_idx, q_idx] * np.abs(pos[p_idx] - pos[q_idx])))

    def optimize_jw_ordering(self) -> np.ndarray:
        """Find a low-cost Jordan-Wigner ordering via spectral ordering + local swap refinement.

        Returns a permutation array π such that orbital π[i] is placed at position i
        in the JW string.
        """
        w = self.majoranapair_weights()[:, :, 0]
        n = self.num_orb

        # ── 1. Spectral ordering (Fiedler vector of weighted Laplacian) ──────
        degree = w.sum(axis=1)
        L = np.diag(degree) - w
        _, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]  # 2nd smallest eigenvector
        perm = np.argsort(fiedler)  # orbital index → JW position

        # ── 2. Local swap refinement ─────────────────────────────────────────
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                swapped = perm.copy()
                swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
                if self.jw_cost(swapped) < self.jw_cost(perm):
                    perm = swapped
                    improved = True

        return perm

    def apply_jw_ordering(
        self, perm: np.ndarray = None, inplace: bool = False
    ) -> "MajoranaPair | None":
        """Permute orbital indices according to a JW ordering.

        :param perm: permutation array where perm[i] is the orbital placed at position i.
                     If None, calls optimize_jw_ordering() to find the optimal permutation.
        :param inplace: if True, modify all tensors in-place and return None;
                        if False, return a new MajoranaPair.
        """
        if perm is None:
            perm = self.optimize_jw_ordering()
        ix = np.ix_(perm, perm)
        ix4 = np.ix_(perm, perm, perm, perm)
        if inplace:
            self.h1e = self.h1e[ix]
            self.h2e = self.h2e[ix4]
            self.f1e = self.f1e[ix]
            self.f2e_diffopp_samespin = self.f2e_diffopp_samespin[ix4]
            self.f2e_diffop_diffspin = self.f2e_diffop_diffspin[ix4]
            self.f2e_sameop_diffspin = self.f2e_sameop_diffspin[ix]
            self._jw_M = None
            self._jw_pauli_labels = None
            self._jw_col_labels = None
            return None
        return MajoranaPair(h0=0, h1e=self.h1e[ix], h2e=self.h2e[ix4])


    def jw_matrix(self):
        """Build (lazily) the linear map M from Majorana coefficients to Pauli coefficients.

        The JW transform is linear: coeffs_pauli = M @ x, where x is the
        concatenation of [f1e.flat, f2e_diffopp_samespin.flat,
        f2e_diffop_diffspin.flat, f2e_sameop_diffspin.flat].

        M is computed once and cached. Subsequent calls return the cached result.

        Returns
        -------
        pauli_labels : list[str]
            Ordered list of Pauli string labels (rows of M).
        col_labels : list[str]
            Human-readable labels for each column of M.
        M : scipy.sparse.csr_matrix, shape (num_pauli_terms, len(x)), complex
        """
        if self._jw_M is not None:
            return self._jw_pauli_labels, self._jw_col_labels, self._jw_M

        n = self.num_orb
        total_qubits = 2 * n

        # Accumulate COO entries
        row_map = {}  # label -> row index
        col_labels = []
        coo_rows, coo_cols, coo_vals = [], [], []

        def _add(lbl, phase, c):
            if lbl not in row_map:
                row_map[lbl] = len(row_map)
            coo_rows.append(row_map[lbl])
            coo_cols.append(c)
            coo_vals.append(phase)

        col = 0

        # ── f1e[p,q]: one Pauli per spin sector ─────────────────────────────
        for p in range(n):
            for q in range(n):
                col_labels.append(f"f1e[{p},{q}]")
                for spin_offset in (0, n):
                    ph, lbl = _jw_pauli(p, q, spin_offset, total_qubits)
                    _add(lbl, complex(ph), col)
                col += 1

        # ── f2e_diffopp_samespin[p,q,r,s]: one Pauli per spin sector ────────
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        col_labels.append(f"f2e_ss[{p},{q},{r},{s}]")
                        for spin_offset in (0, n):
                            ph1, l1 = _jw_pauli(p, q, spin_offset, total_qubits)
                            ph2, l2 = _jw_pauli(r, s, spin_offset, total_qubits)
                            ph, lbl = _multiply_pauli_strings(l1, l2)
                            _add(lbl, complex(ph1 * ph2) * ph, col)
                        col += 1

        # ── f2e_diffop_diffspin[p,q,r,s]: two spin combinations ─────────────
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        col_labels.append(f"f2e_ds[{p},{q},{r},{s}]")
                        for spin_offset in [(0, n), (n, 0)]:
                            ph1, l1 = _jw_pauli(p, q, spin_offset[0], total_qubits)
                            ph2, l2 = _jw_pauli(r, s, spin_offset[1], total_qubits)
                            ph, lbl = _multiply_pauli_strings(l1, l2)
                            _add(lbl, complex(ph1 * ph2) * ph, col)
                        col += 1

        # ── f2e_sameop_diffspin[p,q]: one Pauli ─────────────────────────────
        for p in range(n):
            for q in range(n):
                col_labels.append(f"f2e_so[{p},{q}]")
                ph1, l1 = _jw_pauli(p, q, 0, total_qubits)
                ph2, l2 = _jw_pauli(p, q, n, total_qubits)
                ph, lbl = _multiply_pauli_strings(l1, l2)
                _add(lbl, complex(ph1 * ph2) * ph, col)
                col += 1

        n_rows = len(row_map)
        n_cols = col
        M = scipy.sparse.coo_matrix(
            (coo_vals, (coo_rows, coo_cols)), shape=(n_rows, n_cols), dtype=complex
        ).tocsr()

        self._jw_pauli_labels = list(row_map.keys())
        self._jw_col_labels = col_labels
        self._jw_M = M

        return self._jw_pauli_labels, self._jw_col_labels, self._jw_M

    def jordan_wigner(self) -> PauliHamiltonian:
        pauli_labels, _, M = self.jw_matrix()
        x = np.concatenate(
            [
                self.f1e.ravel(),
                self.f2e_diffopp_samespin.ravel(),
                self.f2e_diffop_diffspin.ravel(),
                self.f2e_sameop_diffspin.ravel(),
            ]
        )
        coeffs = np.array(M @ x).ravel()

        # prepend the constant term
        total_qubits = 2 * self.num_orb
        all_labels = ["I" * total_qubits] + list(pauli_labels)
        all_coeffs = np.concatenate([[self.constant], coeffs])

        return PauliHamiltonian.from_labels_and_weights(all_labels, all_coeffs)


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
