import functools
import numpy as np
from champy.PauliHamiltonian import PauliHamiltonian


@functools.lru_cache(maxsize=16)
def _tz_masks(n: int):
    """Boolean masks for ElectronicStructureTZ._coefficients, cached by n.

    Both masks depend only on the system size n (never on h1e/h2e), so they
    are computed once per unique n and reused across calls — or baked in as
    compile-time constants when the function is run under jax.jit.

    Returns
    -------
    mask_t : (n, n) bool
        Upper-triangular p<q canonical-pair mask. Selects the index range
        for T_pq (and its 3- and 4-index extensions on the (p,q) axes).
    mask_tz : (n, n, n) bool
        Mask for the same-spin TZ term T_pqx Z_rx. Excludes r ∈ {p, q}
        (those configurations would not produce a TZ_same term).
    mask_tt_same : (n, n, n, n) bool
        Mask for the same-spin TT term T_pqx T_rsx with p<q, r<s, p<r,
        and {p,q} ∩ {r,s} = ∅.
    mask_tt_opp : (n, n, n, n) bool
        Mask for the opposite-spin TT term T_pqu T_rsd, p<q and r<s
        (independent (p,q) and (r,s) pairs).
    """
    mask_t = np.triu(np.ones((n, n), dtype=bool), k=1)
    mask_tz = np.ones((n, n, n), dtype=bool)
    for p in range(n):
        mask_tz[p, :, p] = False
        mask_tz[:, p, p] = False
    mask_tt_opp = mask_t[:, :, None, None] & mask_t[None, None, :, :]
    mask_tt_same = mask_tt_opp.copy()
    for p in range(n):
        for q in range(p + 1, n):
            for r in range(n):
                for s in range(r + 1, n):
                    if r <= p or len({p, q} & {r, s}) > 0:
                        mask_tt_same[p, q, r, s] = False
    return mask_t, mask_tz, mask_tt_same, mask_tt_opp


QASM_GATE_DEFS = """\
gate rzz(theta) q0, q1 {
    cx q0, q1;
    rz(theta) q1;
    cx q0, q1;
}
gate givens(theta) q0, q1 {
    s q0;
    ry(pi/2) q1;
    cx q1, q0;
    ry(theta) q0;
    ry(theta) q1;
    cx q1, q0;
    sdg q0;
    ry(-pi/2) q1;
}
gate xyswap q0 {
    rx(pi/2) q0;
    h q0;
    rx(-pi/2) q0;
}
gate cry(theta) c, t {
    sdg t;
    h t;
    rzz(-theta/2) c, t;
    h t;
    s t;
    ry(theta/2) t;
}
gate z_xxyy(theta) qr, q0, q1 {
    s q0;
    ry(pi/2) q1;
    cx q1, q0;
    ry(theta) q0;
    ry(theta) q1;
    cry(-2*theta) qr, q0;
    cry(-2*theta) qr, q1;
    cx q1, q0;
    sdg q0;
    ry(-pi/2) q1;
}
gate xxyy_zbasis q0, q1 {
    s q0;
    ry(pi/2) q1;
    cx q1, q0;
    sdg q0;
    h q0;
    s q0;
    sdg q1;
    h q1;
    s q1;
}
gate xxyy_zbasis_inv q0, q1 {
    sdg q1;
    h q1;
    s q1;
    sdg q0;
    h q0;
    s q0;
    cx q1, q0;
    ry(-pi/2) q1;
    sdg q0;
}
"""


def _qasm_bell(control: int, target: int) -> str:
    """QASM for Bell basis transform: CNOT then H on control."""
    return f"cx q[{control}],q[{target}];\nh q[{control}];\n"


def _qasm_bell_inv(control: int, target: int) -> str:
    """QASM to undo Bell basis transform."""
    return f"h q[{control}];\ncx q[{control}],q[{target}];\n"


def _qasm_parity_tree(qubits: list[int]) -> str:
    """QASM to XOR qubits[1:] onto qubits[0]."""
    s = ""
    for k in qubits[1:]:
        s += f"cx q[{k}],q[{qubits[0]}];\n"
    return s


def _qasm_parity_tree_inv(qubits: list[int]) -> str:
    """QASM to undo _qasm_parity_tree."""
    s = ""
    for k in reversed(qubits[1:]):
        s += f"cx q[{k}],q[{qubits[0]}];\n"
    return s


def _qasm_diagonal_4rot(target: int, qubits: list[int], angles: list[float]) -> str:
    """QASM for exp(i(a1 Z_T Z_A + a2 Z_T Z_A Z_B + a3 Z_T Z_A Z_B Z_C + a4 Z_T Z_A Z_C)/2).

    Accumulates parities of qubits A, B, C onto target T via CNOTs,
    performing Rz after each accumulation step.
    CNOT sequence: A, B, C, B, A, C (6 CNOTs, 4 Rz).
    """
    A, B, C = qubits
    a1, a2, a3, a4 = angles
    return (
        f"cx q[{A}],q[{target}];\n"
        f"rz({a1}) q[{target}];\n"
        f"cx q[{B}],q[{target}];\n"
        f"rz({a2}) q[{target}];\n"
        f"cx q[{C}],q[{target}];\n"
        f"rz({a3}) q[{target}];\n"
        f"cx q[{B}],q[{target}];\n"
        f"rz({a4}) q[{target}];\n"
        f"cx q[{A}],q[{target}];\n"
        f"cx q[{C}],q[{target}];\n"
    )


def _qasm_xy_swap(qubit: int) -> str:
    """QASM for X<->Y swap gate: Rx(-pi/2) H Rx(pi/2). Self-inverse."""
    return f"xyswap q[{qubit}];\n"


class ElectronicStructureTZ:
    """Electronic structure Hamiltonian in the T/Z operator basis.

    Decomposes H into terms built from hopping operators
    T_pqx = a+_px a_qx + a+_qx a_px and number operators
    n_px = a+_px a_px = (I - Z_px) / 2, where p,q are spatial orbitals
    and x is spin (alpha or beta). Each term is particle-number conserving
    and has operator norm 1, making the 1-norm a direct cost metric for
    qDRIFT simulation.

    Operator types (grouped by spin structure):
        Z_px, T_pqx,
        Z_px Z_qx (same-spin), Z_pu Z_qd (opposite-spin),
        T_pqx Z_rx (same-spin, r not in {p,q}), T_pqx Z_ry (opposite-spin),
        T_pqx T_rsx (same-spin, {p,q} ∩ {r,s} = ∅, p < r),
        T_pqu T_rsd (opposite-spin)
    """

    def __init__(self, h0: float, h1e: np.ndarray, h2e: np.ndarray):
        assert h1e.ndim == 2 and h2e.ndim == 4
        assert len(set(h1e.shape)) == 1
        assert set(h1e.shape) == set(h2e.shape)

        n = h1e.shape[0]
        self.num_orb = n
        self.h0 = h0
        self.h1e = h1e
        self.h2e = h2e

        coeffs = ElectronicStructureTZ._coefficients(h1e, h2e)

        self._constant = float(
            h0
            + np.trace(h1e)
            - 0.25 * np.einsum("pqpq->", h2e)
            + 0.5 * np.einsum("ppqq->", h2e)
        )
        self.coeff_Z = coeffs["Z"]
        self.coeff_T = coeffs["T"]
        self.coeff_ZZ_same = coeffs["ZZ_same"]
        self.coeff_ZZ_opp = coeffs["ZZ_opp"]
        self.coeff_TZ_opp = coeffs["TZ_opp"]
        self.coeff_TZ_same = coeffs["TZ_same"]
        self.coeff_TT_same = coeffs["TT_same"]
        self.coeff_TT_opp = coeffs["TT_opp"]

    @staticmethod
    @functools.partial(__import__("jax").jit, static_argnums=())
    def _coefficients(h1e, h2e) -> dict:
        """Build the eight TZ coefficient tensors from h1e, h2e.

        JIT-compiled and JAX-differentiable. Boolean masks are cached by n
        via _tz_masks() and treated as compile-time constants under jax.jit.
        """
        import jax.numpy as jnp

        n = h1e.shape[0]
        mask_t, mask_tz, mask_tt_same, mask_tt_opp = _tz_masks(n)

        coulomb = jnp.einsum("ppqq->pq", h2e)
        exchange = jnp.einsum("pqpq->pq", h2e)
        h_pqrr = jnp.einsum("pqrr->pqr", h2e)
        h_prrq = jnp.einsum("prrq->pqr", h2e)

        # Z_px: 1/2 (1/2 sum_q h_pqpq - h_pp - sum_q h_ppqq) per p
        coeff_Z = 0.5 * (
            0.5 * jnp.einsum("pqpq->p", h2e)
            - jnp.diag(h1e)
            - jnp.einsum("ppqq->p", h2e)
        )

        # T_pqx: (h_pq - 1/2 sum_r h_prrq + sum_r h_pqrr) per p<q
        coeff_T = jnp.where(
            mask_t,
            h1e - 0.5 * jnp.einsum("prrq->pq", h2e) + jnp.einsum("pqrr->pq", h2e),
            0.0,
        )

        # Z_px Z_qx (same-spin): 1/4 (h_ppqq - h_pqpq) per p<q
        coeff_ZZ_same = jnp.where(mask_t, 0.25 * (coulomb - exchange), 0.0)

        # Z_pu Z_qd (opposite-spin): 1/4 h_ppqq per (p, q)
        coeff_ZZ_opp = 0.25 * coulomb

        # TZ opposite-spin: -1/2 h_pqrr, all r
        coeff_TZ_opp = jnp.where(mask_t[:, :, None], -0.5 * h_pqrr, 0.0)

        # TZ same-spin: 1/2 (h_prrq - h_pqrr), r not in {p,q}
        coeff_TZ_same = jnp.where(
            mask_t[:, :, None] & mask_tz, 0.5 * (h_prrq - h_pqrr), 0.0
        )

        # same-spin TT: {p,q} ∩ {r,s} = ∅, canonical order p < r
        coeff_TT_same = jnp.where(mask_tt_same, h2e, 0.0)

        # opposite-spin TT: all (p<q, r<s)
        coeff_TT_opp = jnp.where(mask_tt_opp, h2e, 0.0)

        return {
            "Z": coeff_Z,
            "T": coeff_T,
            "ZZ_same": coeff_ZZ_same,
            "ZZ_opp": coeff_ZZ_opp,
            "TZ_opp": coeff_TZ_opp,
            "TZ_same": coeff_TZ_same,
            "TT_same": coeff_TT_same,
            "TT_opp": coeff_TT_opp,
        }

    @property
    def constant(self) -> float:
        return self._constant

    def __add__(self, other):
        assert isinstance(other, ElectronicStructureTZ) and self.num_orb == other.num_orb
        return ElectronicStructureTZ(
            self.h0 + other.h0, self.h1e + other.h1e, self.h2e + other.h2e
        )

    def __sub__(self, other):
        assert isinstance(other, ElectronicStructureTZ) and self.num_orb == other.num_orb
        return ElectronicStructureTZ(
            self.h0 - other.h0, self.h1e - other.h1e, self.h2e - other.h2e
        )

    def __mul__(self, other):
        assert isinstance(other, (int, float))
        return ElectronicStructureTZ(other * self.h0, other * self.h1e, other * self.h2e)

    __rmul__ = __mul__

    def __eq__(self, other):
        if not isinstance(other, ElectronicStructureTZ):
            return False
        if self.num_orb != other.num_orb:
            return False
        return (
            np.isclose(self.h0, other.h0)
            and np.allclose(self.h1e, other.h1e)
            and np.allclose(self.h2e, other.h2e)
        )

    def one_norm(self) -> float:
        return ElectronicStructureTZ._one_norm(self.h1e, self.h2e)

    @staticmethod
    def _one_norm(h1e: np.ndarray, h2e: np.ndarray) -> float:
        c = ElectronicStructureTZ._coefficients(h1e, h2e)
        return (
            np.sum(np.abs(c["Z"])) * 2
            + np.sum(np.abs(c["T"])) * 2
            + np.sum(np.abs(c["ZZ_same"])) * 2
            + np.sum(np.abs(c["ZZ_opp"]))
            + np.sum(np.abs(c["TZ_opp"])) * 2
            + np.sum(np.abs(c["TZ_same"])) * 2
            + np.sum(np.abs(c["TT_same"])) * 2
            + np.sum(np.abs(c["TT_opp"]))
        )

    @staticmethod
    def _qubit(p: int, x: int, n: int, offset: int) -> int:
        return p - 1 + x * n + offset

    @staticmethod
    def z_circuit_cost() -> int:
        """Entangling gate count for z_circuit (single-qubit Rz)."""
        return 0

    @staticmethod
    def zz_circuit_cost() -> int:
        """Entangling gate count for zz_circuit (one RZZ)."""
        return 1

    @staticmethod
    def t_circuit_cost(p: int, q: int) -> int:
        """Entangling gate count for t_circuit.

        d=1: Givens (2). d>=2: parity tree + xxyy_zbasis + 2 RZZ (2d).
        """
        d = abs(q - p)
        if d == 1:
            return 2
        return 2 * d

    @staticmethod
    def tz_opp_circuit_cost(p: int, q: int) -> int:
        """Entangling gate count for tz_opp_circuit (independent of r).

        d=1: 4. d>=2: 2d+2.
        """
        d = abs(q - p)
        if d == 1:
            return 4
        return 2 * d + 2

    @staticmethod
    def tz_same_circuit_cost(p: int, q: int, r: int) -> int:
        """Entangling gate count for tz_same_circuit.

        r outside [p,q], d=1: 4. r outside, d>=2: 2d+2.
        r inside, d=2: 2 (Givens). r inside, d>=3: 2(d-1).
        """
        d = abs(q - p)
        r_inside = min(p, q) < r < max(p, q)
        if not r_inside:
            return 4 if d == 1 else 2 * d + 2
        return 2 if d == 2 else 2 * (d - 1)

    @staticmethod
    def tt_opp_circuit_cost(p: int, q: int, r: int, s: int) -> int:
        """Entangling gate count for tt_opp_circuit.

        d1=d2=1: xxyy_zbasis + 4 RZZ (8). Otherwise: 2(d1+d2)+6.
        """
        d1 = abs(q - p)
        d2 = abs(s - r)
        if d1 == 1 and d2 == 1:
            return 8
        return 2 * (d1 + d2) + 6

    @staticmethod
    def tt_same_circuit_cost(p: int, q: int, r: int, s: int) -> int:
        """Entangling gate count for same-spin TT (any of nonoverlap, interleaved, nested).

        Dispatches based on the qubit ordering of (p,q) and (r,s):
        - non-overlapping (p<q<r<s or r<s<p<q): 2(d1+d2)+6 (or 8 if d1=d2=1).
        - interleaved (p<r<q<s or r<p<s<q): 10 + extra(|r-p|) + extra(|s-q|),
          which becomes 8 in the minimum case (gaps == 1).
        - nested (p<r<s<q or r<p<q<s): same form as interleaved on inner gaps.
        Here extra(k) = 2k-2 if k>=2 else 0; the minimum case uses xxyy_zbasis (8 gates).
        """
        lo1, hi1 = min(p, q), max(p, q)
        lo2, hi2 = min(r, s), max(r, s)
        if lo1 > lo2:
            lo1, hi1, lo2, hi2 = lo2, hi2, lo1, hi1

        if hi1 < lo2:
            d1, d2 = hi1 - lo1, hi2 - lo2
            if d1 == 1 and d2 == 1:
                return 8
            return 2 * (d1 + d2) + 6
        if hi1 < hi2:
            delta1 = lo2 - lo1
            delta2 = hi2 - hi1
        else:
            delta1 = lo2 - lo1
            delta2 = hi1 - hi2
        if delta1 == 1 and delta2 == 1:
            return 8
        extra = lambda k: 2 * k - 2 if k >= 2 else 0
        return 10 + extra(delta1) + extra(delta2)

    # ── JW ordering ─────────────────────────────────────────────────────────

    def _circuit_cost_tensors(self) -> dict:
        """Lazily build and cache 2-qubit-gate cost tensors keyed by term
        group. Each tensor is indexed by qubit positions: entry [i, j, ...]
        is the cost when the term's orbital indices land at positions
        (i, j, ...) in the JW string. Z and ZZ entries are perm-independent
        scalars.
        """
        if hasattr(self, "_cost_tensors"):
            return self._cost_tensors
        n = self.num_orb
        cost_T = np.zeros((n, n))
        cost_TZ_opp = np.zeros((n, n))
        cost_TZ_same = np.zeros((n, n, n))
        cost_TT_opp = np.zeros((n, n, n, n))
        cost_TT_same = np.zeros((n, n, n, n))
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                cost_T[p, q] = self.t_circuit_cost(p, q)
                cost_TZ_opp[p, q] = self.tz_opp_circuit_cost(p, q)
                for r in range(n):
                    if r != p and r != q:
                        cost_TZ_same[p, q, r] = self.tz_same_circuit_cost(p, q, r)
                    for s in range(n):
                        if s == r:
                            continue
                        cost_TT_opp[p, q, r, s] = self.tt_opp_circuit_cost(p, q, r, s)
                        if len({p, q, r, s}) == 4:
                            cost_TT_same[p, q, r, s] = self.tt_same_circuit_cost(p, q, r, s)
        self._cost_tensors = {
            "Z": self.z_circuit_cost(),
            "ZZ": self.zz_circuit_cost(),
            "T": cost_T,
            "TZ_opp": cost_TZ_opp,
            "TZ_same": cost_TZ_same,
            "TT_opp": cost_TT_opp,
            "TT_same": cost_TT_same,
        }
        return self._cost_tensors

    def jw_cost(self, perm: np.ndarray) -> float:
        """Total 2-qubit-gate cost Σ_α |c_α| · gates_α(perm) under spatial-
        orbital permutation `perm` (perm[i] = orbital placed at JW position
        i within each spin sector).
        """
        n = self.num_orb
        pos = np.empty(n, dtype=int)
        pos[perm] = np.arange(n)
        t = self._circuit_cost_tensors()

        T_p = t["T"][np.ix_(pos, pos)]
        TZopp_p = t["TZ_opp"][np.ix_(pos, pos)]
        TZsame_p = t["TZ_same"][np.ix_(pos, pos, pos)]
        TTopp_p = t["TT_opp"][np.ix_(pos, pos, pos, pos)]
        TTsame_p = t["TT_same"][np.ix_(pos, pos, pos, pos)]

        cost = (
            2.0 * t["Z"] * np.sum(np.abs(self.coeff_Z))
            + 2.0 * t["ZZ"] * np.sum(np.abs(self.coeff_ZZ_same))
            + t["ZZ"] * np.sum(np.abs(self.coeff_ZZ_opp))
            + 2.0 * np.sum(np.abs(self.coeff_T) * T_p)
            + 2.0 * np.sum(np.abs(self.coeff_TZ_opp) * TZopp_p[..., None])
            + 2.0 * np.sum(np.abs(self.coeff_TZ_same) * TZsame_p)
            + np.sum(np.abs(self.coeff_TT_opp) * TTopp_p)
            + 2.0 * np.sum(np.abs(self.coeff_TT_same) * TTsame_p)
        )
        return float(cost)

    def _jw_pair_weights(self) -> np.ndarray:
        """Symmetric n×n weight matrix aggregating coefficient magnitudes for
        orbital pairs (p,q) appearing as hopping pairs. Used to seed the
        spectral ordering in optimize_jw_ordering."""
        w = np.abs(self.coeff_T).copy()
        w += np.einsum("pqr->pq", np.abs(self.coeff_TZ_opp))
        w += np.einsum("pqr->pq", np.abs(self.coeff_TZ_same))
        w += np.einsum("pqrs->pq", np.abs(self.coeff_TT_opp))
        w += np.einsum("pqrs->rs", np.abs(self.coeff_TT_opp))
        w += np.einsum("pqrs->pq", np.abs(self.coeff_TT_same))
        w += np.einsum("pqrs->rs", np.abs(self.coeff_TT_same))
        return w + w.T

    def optimize_jw_ordering(self) -> np.ndarray:
        """Find a low-cost Jordan-Wigner ordering via spectral seeding +
        adjacent-swap refinement against `jw_cost`. Returns a permutation
        array π of length num_orb where π[i] is the spatial orbital placed
        at JW position i (within each spin sector)."""
        w = self._jw_pair_weights()
        n = self.num_orb

        # ── 1. Spectral seed (Fiedler vector of weighted Laplacian) ─────────
        degree = w.sum(axis=1)
        L = np.diag(degree) - w
        _, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
        perm = np.argsort(fiedler)

        # ── 2. Adjacent-swap refinement against actual jw_cost ──────────────
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
    ) -> "ElectronicStructureTZ | None":
        """Permute spatial orbital indices according to a JW ordering.

        :param perm: permutation array where perm[i] is the orbital placed at
                     position i. If None, calls optimize_jw_ordering().
        :param inplace: if True, rebuild this instance in-place and return None;
                        if False, return a new ElectronicStructureTZ.
        """
        if perm is None:
            perm = self.optimize_jw_ordering()
        ix = np.ix_(perm, perm)
        ix4 = np.ix_(perm, perm, perm, perm)
        h1e_p = self.h1e[ix]
        h2e_p = self.h2e[ix4]
        if inplace:
            self.__init__(h0=self.h0, h1e=h1e_p, h2e=h2e_p)
            return None
        return ElectronicStructureTZ(h0=self.h0, h1e=h1e_p, h2e=h2e_p)

    def plot_orbital_graph(self, optimize_jw: bool = False) -> None:
        """Plot the orbital graph using a spring layout, optionally with JW
        orderings overlaid as paths.

        Vertex weight: |coeff_Z[p]|. Edge weight: aggregated hopping-pair
        magnitude from _jw_pair_weights() (same matrix that seeds
        optimize_jw_ordering). Heavier edges pull nodes closer in the layout.

        :param optimize_jw: if True, overlay the optimized JW ordering in
                            addition to the default identity ordering.
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        import matplotlib.colors as mcolors
        import matplotlib.colorbar as mcolorbar

        n = self.num_orb
        w = self._jw_pair_weights()
        diag_vals = np.abs(self.coeff_Z)

        cmap = plt.colormaps["Blues"]
        nonzero = w[w > 0]
        vmin = nonzero.min() if nonzero.size > 0 else 1e-12
        vmax = max(w.max(), diag_vals.max(), vmin * 10)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

        # Build graph with edge weights and compute layout
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for p in range(n):
            for q in range(p + 1, n):
                if w[p, q] > 1e-6 * w.max():
                    G.add_edge(p, q, weight=w[p, q])
        pos = nx.spring_layout(G, weight="weight", seed=42)

        # JW orderings to display
        jw_orderings = [("JW default", np.arange(n), "red")]
        if optimize_jw:
            jw_orderings.append(("JW optimized", self.optimize_jw_ordering(), "green"))

        n_cols = 1 + len(jw_orderings)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        fig.subplots_adjust(right=0.88)

        def _draw_vertices(ax):
            for p in range(n):
                circle = plt.Circle(
                    pos[p],
                    0.07,
                    facecolor=cmap(norm(diag_vals[p])) if diag_vals[p] > 0 else "lightgrey",
                    edgecolor="black",
                    linewidth=1.5,
                    zorder=3,
                )
                ax.add_patch(circle)
                if diag_vals[p] > 0:
                    r, g, b, _ = cmap(norm(diag_vals[p]))
                    luminance = 0.299 * r + 0.587 * g + 0.114 * b
                    text_color = "white" if luminance < 0.5 else "black"
                else:
                    text_color = "black"
                ax.text(
                    *pos[p],
                    str(p),
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    zorder=4,
                    color=text_color,
                )
            ax.set_aspect("equal")
            ax.axis("off")
            ax.autoscale_view()

        # ── Left: orbital interaction graph ──────────────────────────────────
        ax = axes[0]
        for p, q in G.edges():
            xs = [pos[p][0], pos[q][0]]
            ys = [pos[p][1], pos[q][1]]
            ax.plot(xs, ys, color=cmap(norm(w[p, q])), lw=2, zorder=1)
        _draw_vertices(ax)
        ax.set_title(
            "Orbital graph\nvertex: $|c_{Z_p}|$,  edge: hopping-pair weight",
            fontsize=10,
        )

        # ── Right: one subplot per JW ordering ───────────────────────────────
        for ax, (title, perm, color) in zip(axes[1:], jw_orderings):
            cost = self.jw_cost(perm)
            for i in range(n - 1):
                xs = [pos[perm[i]][0], pos[perm[i + 1]][0]]
                ys = [pos[perm[i]][1], pos[perm[i + 1]][1]]
                ax.plot(xs, ys, color=color, lw=2, zorder=1)
            _draw_vertices(ax)
            ax.set_title(f"{title}\ncost = {cost:.3f}", fontsize=10)

        # Shared colorbar
        cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        mcolorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
        cax.set_title(r"$w$", fontsize=10)

        plt.show()

    @staticmethod
    def z_circuit(p: int, x: int, angle: float, n: int, offset: int = 0) -> str:
        """QASM circuit for exp(i * angle * Z_px)."""
        q = ElectronicStructureTZ._qubit(p, x, n, offset)
        return f"rz({-2 * angle}) q[{q}];\n"

    @staticmethod
    def zz_circuit(
        p: int, x_p: int, q: int, x_q: int, angle: float, n: int, offset: int = 0
    ) -> str:
        """QASM circuit for exp(i * angle * Z_px Z_qy)."""
        q0 = ElectronicStructureTZ._qubit(p, x_p, n, offset)
        q1 = ElectronicStructureTZ._qubit(q, x_q, n, offset)
        return f"rzz({-2 * angle}) q[{q0}],q[{q1}];\n"

    @staticmethod
    def t_circuit(
        p: int, q: int, x: int, angle: float, n: int, offset: int = 0
    ) -> str:
        """QASM circuit for exp(i * angle * T_pqx), where T_pq = (XX+YY)/2.

        d=1: Givens gate (2 entangling gates).
        d>=2: parity tree + z_xxyy (2d entangling gates).
        Requires QASM_GATE_DEFS to be prepended to the program.
        """
        _q = ElectronicStructureTZ._qubit
        qp = _q(p, x, n, offset)
        qq = _q(q, x, n, offset)
        d = abs(qq - qp)
        if d == 1:
            return f"givens({angle}) q[{qp}],q[{qq}];\n"

        sign = 1 if qq > qp else -1
        inter = [qp + sign * i for i in range(1, d)]
        m = inter[0]

        s = ""
        s += _qasm_parity_tree(inter)
        s += f"xxyy_zbasis q[{qp}],q[{qq}];\n"
        s += f"rzz({angle}) q[{m}],q[{qp}];\n"
        s += f"rzz({angle}) q[{m}],q[{qq}];\n"
        s += f"xxyy_zbasis_inv q[{qp}],q[{qq}];\n"
        s += _qasm_parity_tree_inv(inter)
        return s

    @staticmethod
    def tz_opp_circuit(
        p: int, q: int, x: int, r: int, angle: float, n: int, offset: int = 0
    ) -> str:
        """QASM circuit for exp(i * angle * T_pqx Z_ry) with y != x.

        d=1: xxyy_zbasis + 2 RZZ on (r, p) and (r, q) (4 entangling gates).
        d>=2: parity tree + Z_r XOR + xxyy_zbasis + 2 RZZ + undo (2d+2 entangling gates).
        """
        _q = ElectronicStructureTZ._qubit
        qp = _q(p, x, n, offset)
        qq = _q(q, x, n, offset)
        qr = _q(r, 1 - x, n, offset)
        d = abs(qq - qp)
        if d == 1:
            s = ""
            s += f"xxyy_zbasis q[{qp}],q[{qq}];\n"
            s += f"rzz({angle}) q[{qr}],q[{qp}];\n"
            s += f"rzz({angle}) q[{qr}],q[{qq}];\n"
            s += f"xxyy_zbasis_inv q[{qp}],q[{qq}];\n"
            return s

        sign = 1 if qq > qp else -1
        inter = [qp + sign * i for i in range(1, d)]
        m = inter[0]

        s = ""
        s += _qasm_parity_tree(inter)
        s += f"cx q[{qr}],q[{m}];\n"
        s += f"xxyy_zbasis q[{qp}],q[{qq}];\n"
        s += f"rzz({angle}) q[{m}],q[{qp}];\n"
        s += f"rzz({angle}) q[{m}],q[{qq}];\n"
        s += f"xxyy_zbasis_inv q[{qp}],q[{qq}];\n"
        s += f"cx q[{qr}],q[{m}];\n"
        s += _qasm_parity_tree_inv(inter)
        return s

    @staticmethod
    def tz_same_circuit(
        p: int, q: int, r: int, x: int, angle: float, n: int, offset: int = 0
    ) -> str:
        """QASM circuit for exp(i * angle * T_pqx Z_rx) with r not in {p,q}.

        Cases:
        - r outside [p,q], d=1: xxyy_zbasis + 2 RZZ on (r, p) and (r, q) (4 entangling gates).
        - r outside [p,q], d>=2: parity tree + Z_r XOR + xxyy_zbasis + 2 RZZ (2d+2 entangling gates).
        - r inside (p,q), d=2: Givens gate (2 entangling gates).
        - r inside (p,q), d>=3: parity tree excluding r + xxyy_zbasis + 2 RZZ (2(d-1) entangling gates).
        """
        _q = ElectronicStructureTZ._qubit
        qp = _q(p, x, n, offset)
        qq = _q(q, x, n, offset)
        qr = _q(r, x, n, offset)
        d = abs(qq - qp)
        sign = 1 if qq > qp else -1
        r_inside = (min(qp, qq) < qr < max(qp, qq))

        if not r_inside:
            if d == 1:
                s = ""
                s += f"xxyy_zbasis q[{qp}],q[{qq}];\n"
                s += f"rzz({angle}) q[{qr}],q[{qp}];\n"
                s += f"rzz({angle}) q[{qr}],q[{qq}];\n"
                s += f"xxyy_zbasis_inv q[{qp}],q[{qq}];\n"
                return s
            inter = [qp + sign * i for i in range(1, d)]
            m = inter[0]
            s = ""
            s += _qasm_parity_tree(inter)
            s += f"cx q[{qr}],q[{m}];\n"
            s += f"xxyy_zbasis q[{qp}],q[{qq}];\n"
            s += f"rzz({angle}) q[{m}],q[{qp}];\n"
            s += f"rzz({angle}) q[{m}],q[{qq}];\n"
            s += f"xxyy_zbasis_inv q[{qp}],q[{qq}];\n"
            s += f"cx q[{qr}],q[{m}];\n"
            s += _qasm_parity_tree_inv(inter)
            return s

        # r inside (p,q): Z-string includes Z_r, so T_pq Z_r cancels Z_r
        if d == 2:
            return f"givens({angle}) q[{qp}],q[{qq}];\n"

        # d>=3: parity tree over intermediates excluding r
        inter = [qp + sign * i for i in range(1, d) if qp + sign * i != qr]
        m = inter[0]
        s = ""
        s += _qasm_parity_tree(inter)
        s += f"xxyy_zbasis q[{qp}],q[{qq}];\n"
        s += f"rzz({angle}) q[{m}],q[{qp}];\n"
        s += f"rzz({angle}) q[{m}],q[{qq}];\n"
        s += f"xxyy_zbasis_inv q[{qp}],q[{qq}];\n"
        s += _qasm_parity_tree_inv(inter)
        return s

    @staticmethod
    def tt_opp_circuit(
        p: int, q: int, r: int, s_orb: int, angle: float, n: int, offset: int = 0
    ) -> str:
        """QASM circuit for exp(i * angle * T_pqu T_rsd).

        d1=d2=1: xxyy_zbasis on both pairs + 4 RZZ (8 entangling gates).
        d1>=2 or d2>=2: Bell + parity tree + diagonal_4rot (2(d1+d2)+6 entangling).
        """
        _q = ElectronicStructureTZ._qubit
        qp = _q(p, 0, n, offset)
        qq = _q(q, 0, n, offset)
        qr = _q(r, 1, n, offset)
        qs = _q(s_orb, 1, n, offset)
        d1 = abs(qq - qp)
        d2 = abs(qs - qr)

        if d1 == 1 and d2 == 1:
            s = ""
            s += f"xxyy_zbasis q[{qp}],q[{qq}];\n"
            s += f"xxyy_zbasis q[{qr}],q[{qs}];\n"
            s += f"rzz({-angle/2}) q[{qp}],q[{qr}];\n"
            s += f"rzz({-angle/2}) q[{qp}],q[{qs}];\n"
            s += f"rzz({-angle/2}) q[{qq}],q[{qr}];\n"
            s += f"rzz({-angle/2}) q[{qq}],q[{qs}];\n"
            s += f"xxyy_zbasis_inv q[{qr}],q[{qs}];\n"
            s += f"xxyy_zbasis_inv q[{qp}],q[{qq}];\n"
            return s

        sign1 = 1 if qq > qp else -1
        sign2 = 1 if qs > qr else -1
        inter1 = [qp + sign1 * i for i in range(1, d1)]
        inter2 = [qr + sign2 * i for i in range(1, d2)]
        m1 = inter1[0]
        m2 = inter2[0]
        a = angle / 4

        s = ""
        s += _qasm_parity_tree(inter1)
        s += _qasm_parity_tree(inter2)
        s += _qasm_bell(qp, qq)
        s += _qasm_bell(qr, qs)
        # Add parities
        s += f"cx q[{m1}],q[{qp}];\n"
        s += f"cx q[{m2}],q[{qr}];\n"
        s += _qasm_diagonal_4rot(
            qp, [qr, qs, qq], [-2*a, 2*a, -2*a, 2*a]
        )
        # Undo parities
        s += f"cx q[{m2}],q[{qr}];\n"
        s += f"cx q[{m1}],q[{qp}];\n"
        s += _qasm_bell_inv(qr, qs)
        s += _qasm_bell_inv(qp, qq)
        s += _qasm_parity_tree_inv(inter2)
        s += _qasm_parity_tree_inv(inter1)
        return s

    @staticmethod
    def tt_same_nonoverlap_circuit(
        p: int, q: int, r: int, s_orb: int, x: int, angle: float, n: int, offset: int = 0
    ) -> str:
        """QASM circuit for exp(i * angle * T_pqx T_rsx), non-overlapping case.

        Requires p<q<r<s (or r<s<p<q). Same structure as opposite-spin TT.
        d1=d2=1: xxyy_zbasis on both pairs + 4 RZZ (8 entangling).
        Otherwise: Bell + parity tree + diagonal_4rot (2(d1+d2)+6 entangling).
        """
        _q = ElectronicStructureTZ._qubit
        qp = _q(p, x, n, offset)
        qq = _q(q, x, n, offset)
        qr = _q(r, x, n, offset)
        qs = _q(s_orb, x, n, offset)
        d1 = abs(qq - qp)
        d2 = abs(qs - qr)

        if d1 == 1 and d2 == 1:
            s = ""
            s += f"xxyy_zbasis q[{qp}],q[{qq}];\n"
            s += f"xxyy_zbasis q[{qr}],q[{qs}];\n"
            s += f"rzz({-angle/2}) q[{qp}],q[{qr}];\n"
            s += f"rzz({-angle/2}) q[{qp}],q[{qs}];\n"
            s += f"rzz({-angle/2}) q[{qq}],q[{qr}];\n"
            s += f"rzz({-angle/2}) q[{qq}],q[{qs}];\n"
            s += f"xxyy_zbasis_inv q[{qr}],q[{qs}];\n"
            s += f"xxyy_zbasis_inv q[{qp}],q[{qq}];\n"
            return s

        sign1 = 1 if qq > qp else -1
        sign2 = 1 if qs > qr else -1
        inter1 = [qp + sign1 * i for i in range(1, d1)]
        inter2 = [qr + sign2 * i for i in range(1, d2)]
        a = angle / 4

        s = ""
        s += _qasm_parity_tree(inter1)
        s += _qasm_parity_tree(inter2)
        s += _qasm_bell(qp, qq)
        s += _qasm_bell(qr, qs)
        # Add parities
        if len(inter1) > 0:
            s += f"cx q[{inter1[0]}],q[{qp}];\n"
        if len(inter2) > 0:
            s += f"cx q[{inter2[0]}],q[{qr}];\n"
        s += _qasm_diagonal_4rot(
            qp, [qr, qs, qq], [-2*a, 2*a, -2*a, 2*a]
        )
        # Undo parities
        if len(inter2) > 0:
            s += f"cx q[{inter2[0]}],q[{qr}];\n"
        if len(inter1) > 0:
            s += f"cx q[{inter1[0]}],q[{qp}];\n"
        s += _qasm_bell_inv(qr, qs)
        s += _qasm_bell_inv(qp, qq)
        s += _qasm_parity_tree_inv(inter2)
        s += _qasm_parity_tree_inv(inter1)
        return s

    @staticmethod
    def tt_same_interleaved_circuit(
        p: int, q: int, r: int, s_orb: int, x: int, angle: float, n: int, offset: int = 0
    ) -> str:
        """QASM circuit for exp(i * angle * T_pqx T_rsx), overlapping p<r<q<s or r<p<s<q.

        X<->Y swap on the two inner qubits aligns (p,q) and (r,s) as
        simultaneous XX/YY. Z-strings partially cancel, leaving
        intermediates a+1..c-1 and b+1..d-1 where a<c<b<d.
        Minimum case (no outer intermediates): xxyy_zbasis + 4 RZZ (8 entangling).
        """
        _q = ElectronicStructureTZ._qubit
        qp = _q(p, x, n, offset)
        qq = _q(q, x, n, offset)
        qr = _q(r, x, n, offset)
        qs = _q(s_orb, x, n, offset)
        # Identify interleaved ordering: a < c < b < d
        if qp < qr:
            qa, qb, qc, qd = qp, qq, qr, qs
        else:
            qa, qb, qc, qd = qr, qs, qp, qq
        inter1 = list(range(qa + 1, qc))
        inter2 = list(range(qb + 1, qd))
        a = angle / 4

        if len(inter1) == 0 and len(inter2) == 0:
            s = ""
            s += _qasm_xy_swap(qa)
            s += _qasm_xy_swap(qc)
            s += f"xxyy_zbasis q[{qa}],q[{qb}];\n"
            s += f"xxyy_zbasis q[{qc}],q[{qd}];\n"
            # target = (Z_b Z_c - Z_a Z_c - Z_b Z_d + Z_a Z_d)/4
            s += f"rzz({-angle/2}) q[{qa}],q[{qd}];\n"   # +Z_a Z_d
            s += f"rzz({angle/2}) q[{qa}],q[{qc}];\n"    # -Z_a Z_c
            s += f"rzz({angle/2}) q[{qb}],q[{qd}];\n"    # -Z_b Z_d
            s += f"rzz({-angle/2}) q[{qb}],q[{qc}];\n"   # +Z_b Z_c
            s += f"xxyy_zbasis_inv q[{qc}],q[{qd}];\n"
            s += f"xxyy_zbasis_inv q[{qa}],q[{qb}];\n"
            s += _qasm_xy_swap(qc)
            s += _qasm_xy_swap(qa)
            return s

        s = ""
        s += _qasm_xy_swap(qa)
        s += _qasm_xy_swap(qc)
        s += _qasm_parity_tree(inter1)
        s += _qasm_parity_tree(inter2)
        s += _qasm_bell(qa, qb)
        s += _qasm_bell(qc, qd)
        # Add parities
        if len(inter1) > 0:
            s += f"cx q[{inter1[0]}],q[{qa}];\n"
        if len(inter2) > 0:
            s += f"cx q[{inter2[0]}],q[{qc}];\n"
        s += _qasm_diagonal_4rot(
            qa, [qc, qb, qd], [2*a, 2*a, 2*a, 2*a]
        )
        # Undo parities
        if len(inter2) > 0:
            s += f"cx q[{inter2[0]}],q[{qc}];\n"
        if len(inter1) > 0:
            s += f"cx q[{inter1[0]}],q[{qa}];\n"
        s += _qasm_bell_inv(qc, qd)
        s += _qasm_bell_inv(qa, qb)
        s += _qasm_parity_tree_inv(inter2)
        s += _qasm_parity_tree_inv(inter1)
        s += _qasm_xy_swap(qc)
        s += _qasm_xy_swap(qa)
        return s

    @staticmethod
    def tt_same_nested_circuit(
        p: int, q: int, r: int, s_orb: int, x: int, angle: float, n: int, offset: int = 0
    ) -> str:
        """QASM circuit for exp(i * angle * T_pqx T_rsx), nested case p<r<s<q or r<p<q<s.

        The outer pair's Z-string contains both inner qubits. Z-strings cancel
        in the overlap region, leaving segments a+1..b-1 and c+1..d-1 where
        (a,d) is the outer pair and (b,c) is the inner pair. Both parity trees
        add to qubit a. No X<->Y swap needed (pairs already matched as XX/YY).
        Minimum case (no outer intermediates): xxyy_zbasis + 4 RZZ (8 entangling).
        """
        _q = ElectronicStructureTZ._qubit
        qp = _q(p, x, n, offset)
        qq = _q(q, x, n, offset)
        qr = _q(r, x, n, offset)
        qs = _q(s_orb, x, n, offset)
        # Identify nesting: outer (a,d) contains inner (b,c), a<b<c<d
        if qp < qr:
            qa, qb, qc, qd = qp, qr, qs, qq
        else:
            qa, qb, qc, qd = qr, qp, qq, qs
        inter1 = list(range(qa + 1, qb))
        inter2 = list(range(qc + 1, qd))
        a = angle / 4

        if len(inter1) == 0 and len(inter2) == 0:
            s = ""
            s += f"xxyy_zbasis q[{qa}],q[{qd}];\n"
            s += f"xxyy_zbasis q[{qb}],q[{qc}];\n"
            # target = -(Z_a + Z_d)(Z_b + Z_c)/4 = -(Z_a Z_b + Z_a Z_c + Z_b Z_d + Z_c Z_d)/4
            s += f"rzz({angle/2}) q[{qa}],q[{qb}];\n"
            s += f"rzz({angle/2}) q[{qa}],q[{qc}];\n"
            s += f"rzz({angle/2}) q[{qb}],q[{qd}];\n"
            s += f"rzz({angle/2}) q[{qc}],q[{qd}];\n"
            s += f"xxyy_zbasis_inv q[{qb}],q[{qc}];\n"
            s += f"xxyy_zbasis_inv q[{qa}],q[{qd}];\n"
            return s

        s = ""
        s += _qasm_parity_tree(inter1)
        s += _qasm_parity_tree(inter2)
        s += _qasm_bell(qa, qd)
        s += _qasm_bell(qb, qc)
        # Add parities (both to qa)
        if len(inter1) > 0:
            s += f"cx q[{inter1[0]}],q[{qa}];\n"
        if len(inter2) > 0:
            s += f"cx q[{inter2[0]}],q[{qa}];\n"
        s += _qasm_diagonal_4rot(
            qa, [qb, qc, qd], [2*a, -2*a, 2*a, -2*a]
        )
        # Undo parities
        if len(inter2) > 0:
            s += f"cx q[{inter2[0]}],q[{qa}];\n"
        if len(inter1) > 0:
            s += f"cx q[{inter1[0]}],q[{qa}];\n"
        s += _qasm_bell_inv(qb, qc)
        s += _qasm_bell_inv(qa, qd)
        s += _qasm_parity_tree_inv(inter2)
        s += _qasm_parity_tree_inv(inter1)
        return s

    def to_pauli_hamiltonian(self) -> PauliHamiltonian:
        n = self.num_orb
        nq = 2 * n  # alpha qubits: 0..n-1, beta qubits: n..2n-1
        labels = []
        weights = []

        def _pauli_label(ops: dict) -> str:
            """Build a Pauli label string from {qubit_index: 'X'|'Y'|'Z'} dict."""
            chars = ["I"] * nq
            for q, p in ops.items():
                chars[q] = p
            return "".join(chars)

        def _t_labels(p: int, q: int, spin_offset: int):
            """Return the two Pauli strings for T_pq on a given spin sector.
            T_pq = (X_p Z_{p+1}...Z_{q-1} X_q + Y_p Z_{p+1}...Z_{q-1} Y_q) / 2
            with p < q, qubits offset by spin_offset.
            """
            ops_xx = {}
            ops_yy = {}
            pp, qq = p + spin_offset, q + spin_offset
            ops_xx[pp] = "X"
            ops_yy[pp] = "Y"
            for k in range(pp + 1, qq):
                ops_xx[k] = "Z"
                ops_yy[k] = "Z"
            ops_xx[qq] = "X"
            ops_yy[qq] = "Y"
            return _pauli_label(ops_xx), _pauli_label(ops_yy)

        # Identity
        labels.append("I" * nq)
        weights.append(self._constant)

        # Z_px terms
        for p in range(n):
            c = self.coeff_Z[p]
            if c != 0:
                for spin_offset in [0, n]:
                    labels.append(_pauli_label({p + spin_offset: "Z"}))
                    weights.append(c)

        # T_pqx terms
        for p in range(n):
            for q in range(p + 1, n):
                c = self.coeff_T[p, q]
                if c == 0:
                    continue
                for spin_offset in [0, n]:
                    lxx, lyy = _t_labels(p, q, spin_offset)
                    labels.append(lxx)
                    weights.append(c / 2)
                    labels.append(lyy)
                    weights.append(c / 2)

        # Z_px Z_qx (same-spin)
        for p in range(n):
            for q in range(p + 1, n):
                c = self.coeff_ZZ_same[p, q]
                if c == 0:
                    continue
                for spin_offset in [0, n]:
                    labels.append(
                        _pauli_label(
                            {p + spin_offset: "Z", q + spin_offset: "Z"}
                        )
                    )
                    weights.append(c)

        # Z_pu Z_qd (opposite-spin)
        for p in range(n):
            for q in range(n):
                c = self.coeff_ZZ_opp[p, q]
                if c == 0:
                    continue
                labels.append(_pauli_label({p: "Z", q + n: "Z"}))
                weights.append(c)

        # T_pqx Z_ry (opposite-spin, x!=y)
        for p in range(n):
            for q in range(p + 1, n):
                for r in range(n):
                    c = self.coeff_TZ_opp[p, q, r]
                    if c == 0:
                        continue
                    # x=alpha, y=beta: T_pq,alpha * Z_r,beta
                    lxx, lyy = _t_labels(p, q, 0)
                    z_qubit = r + n
                    lxx_z = list(lxx)
                    lxx_z[z_qubit] = "Z"
                    lyy_z = list(lyy)
                    lyy_z[z_qubit] = "Z"
                    labels.append("".join(lxx_z))
                    weights.append(c / 2)
                    labels.append("".join(lyy_z))
                    weights.append(c / 2)
                    # x=beta, y=alpha: T_pq,beta * Z_r,alpha
                    lxx, lyy = _t_labels(p, q, n)
                    z_qubit = r
                    lxx_z = list(lxx)
                    lxx_z[z_qubit] = "Z"
                    lyy_z = list(lyy)
                    lyy_z[z_qubit] = "Z"
                    labels.append("".join(lxx_z))
                    weights.append(c / 2)
                    labels.append("".join(lyy_z))
                    weights.append(c / 2)

        # T_pqx Z_rx (same-spin, r not in {p,q})
        for p in range(n):
            for q in range(p + 1, n):
                for r in range(n):
                    c = self.coeff_TZ_same[p, q, r]
                    if c == 0:
                        continue
                    for spin_offset in [0, n]:
                        lxx, lyy = _t_labels(p, q, spin_offset)
                        z_qubit = r + spin_offset
                        z_label = _pauli_label({z_qubit: "Z"})
                        lxx_new, phase_xx = _multiply_labels(lxx, z_label)
                        lyy_new, phase_yy = _multiply_labels(lyy, z_label)
                        labels.append(lxx_new)
                        weights.append(c / 2 * phase_xx)
                        labels.append(lyy_new)
                        weights.append(c / 2 * phase_yy)

        # T_pqx T_rsx (same-spin, {p,q}∩{r,s}=∅, p<r)
        for p in range(n):
            for q in range(p + 1, n):
                for r in range(n):
                    for s in range(r + 1, n):
                        c = self.coeff_TT_same[p, q, r, s]
                        if c == 0:
                            continue
                        for spin_offset in [0, n]:
                            lxx1, lyy1 = _t_labels(p, q, spin_offset)
                            lxx2, lyy2 = _t_labels(r, s, spin_offset)
                            for l1, l2 in [
                                (lxx1, lxx2),
                                (lxx1, lyy2),
                                (lyy1, lxx2),
                                (lyy1, lyy2),
                            ]:
                                prod, phase = _multiply_labels(l1, l2)
                                labels.append(prod)
                                weights.append(c / 4 * phase)

        # T_pqu T_rsd (opposite-spin)
        for p in range(n):
            for q in range(p + 1, n):
                for r in range(n):
                    for s in range(r + 1, n):
                        c = self.coeff_TT_opp[p, q, r, s]
                        if c == 0:
                            continue
                        lxx1, lyy1 = _t_labels(p, q, 0)  # alpha
                        lxx2, lyy2 = _t_labels(r, s, n)  # beta
                        # different spin sectors, no overlap
                        for l1, l2 in [
                            (lxx1, lxx2),
                            (lxx1, lyy2),
                            (lyy1, lxx2),
                            (lyy1, lyy2),
                        ]:
                            prod, phase = _multiply_labels(l1, l2)
                            labels.append(prod)
                            weights.append(c / 4 * phase)

        return PauliHamiltonian.from_labels_and_weights(labels, weights)


def _multiply_labels(l1: str, l2: str):
    """Multiply two Pauli label strings, return (result_label, phase)."""
    phase = 1.0
    chars = []
    for a, b in zip(l1, l2):
        p, ph = _pauli_mult(a, b)
        chars.append(p)
        phase *= ph
    return "".join(chars), phase


def _pauli_mult(a: str, b: str):
    """Single-qubit Pauli multiplication, return (result, phase)."""
    if a == "I":
        return b, 1.0
    if b == "I":
        return a, 1.0
    if a == b:
        return "I", 1.0
    # XY=iZ, YX=-iZ, XZ=-iY, ZX=iY, YZ=iX, ZY=-iX
    table = {
        ("X", "Y"): ("Z", 1j),
        ("Y", "X"): ("Z", -1j),
        ("X", "Z"): ("Y", -1j),
        ("Z", "X"): ("Y", 1j),
        ("Y", "Z"): ("X", 1j),
        ("Z", "Y"): ("X", -1j),
    }
    return table[(a, b)]
