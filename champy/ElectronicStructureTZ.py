import functools
import numpy as np
from champy import circuits
from champy.ElectronicStructure import ElectronicStructure
from champy.PauliHamiltonian import PauliHamiltonian, multiply_labels


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


class ElectronicStructureTZ(ElectronicStructure):
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

    def __init__(self, h0: float, h1e: np.ndarray, h2e: np.ndarray, num_elec: int):
        super().__init__(h0=h0, h1e=h1e, h2e=h2e, num_elec=num_elec)

        coeffs = ElectronicStructureTZ._coefficients(h1e, h2e)

        self._constant = float(
            h0
            + np.trace(h1e)
            - 0.25 * np.einsum("pqpq->", h2e)
            + 0.5 * np.einsum("ppqq->", h2e)
        )
        self.coeff_Z = np.array(coeffs["Z"])
        self.coeff_T = np.array(coeffs["T"])
        self.coeff_ZZ_same = np.array(coeffs["ZZ_same"])
        self.coeff_ZZ_opp = np.array(coeffs["ZZ_opp"])
        self.coeff_TZ_opp = np.array(coeffs["TZ_opp"])
        self.coeff_TZ_same = np.array(coeffs["TZ_same"])
        self.coeff_TT_same = np.array(coeffs["TT_same"])
        self.coeff_TT_opp = np.array(coeffs["TT_opp"])

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

    def one_norm(self) -> float:
        c = {
            "Z": self.coeff_Z,
            "T": self.coeff_T,
            "ZZ_same": self.coeff_ZZ_same,
            "ZZ_opp": self.coeff_ZZ_opp,
            "TZ_opp": self.coeff_TZ_opp,
            "TZ_same": self.coeff_TZ_same,
            "TT_same": self.coeff_TT_same,
            "TT_opp": self.coeff_TT_opp,
        }
        return float(ElectronicStructureTZ._one_norm_from_coeffs(np, c))

    @staticmethod
    def _one_norm(h1e, h2e):
        """JAX-differentiable 1-norm from integrals, for use in optimizers."""
        import jax.numpy as jnp

        c = ElectronicStructureTZ._coefficients(h1e, h2e)
        return ElectronicStructureTZ._one_norm_from_coeffs(jnp, c)

    @staticmethod
    def _one_norm_from_coeffs(xp, c):
        """Multiplicity-weighted sum of |c| over the 8 TZ term groups.

        Spin-pair conventions: same-spin terms count ×2 (alpha+beta);
        opposite-spin Z·Z and T·T terms count ×1 (one stored coeff per
        unordered spin pair); opposite-spin T·Z counts ×2 (the T-spin
        index distinguishes the two orderings).
        """
        return (
            xp.sum(xp.abs(c["Z"])) * 2
            + xp.sum(xp.abs(c["T"])) * 2
            + xp.sum(xp.abs(c["ZZ_same"])) * 2
            + xp.sum(xp.abs(c["ZZ_opp"]))
            + xp.sum(xp.abs(c["TZ_opp"])) * 2
            + xp.sum(xp.abs(c["TZ_same"])) * 2
            + xp.sum(xp.abs(c["TT_same"])) * 2
            + xp.sum(xp.abs(c["TT_opp"]))
        )

    # ── JW ordering ─────────────────────────────────────────────────────────

    def _circuit_cost_tensors(
        self, *, cost_zz: float = 1.0, cost_1q: float = 0.0
    ) -> dict:
        """Lazily build and cache per-term-group cost tensors. Each tensor is
        indexed by qubit positions: entry [i, j, ...] is the cost when the
        term's orbital indices land at positions (i, j, ...) in the JW string.
        Z and ZZ entries are perm-independent scalars.

        Cost weights are forwarded to the underlying ``circuits.*_circuit_cost``
        functions. Results are cached per ``(cost_zz, cost_1q)`` pair on the
        instance; the cache survives orbital permutation since cost tensors
        depend only on ``num_orb``.
        """
        key = (cost_zz, cost_1q)
        if not hasattr(self, "_cost_tensors_cache"):
            self._cost_tensors_cache = {}
        if key in self._cost_tensors_cache:
            return self._cost_tensors_cache[key]
        n = self.num_orb
        cost_T = np.zeros((n, n))
        cost_TZ_opp = np.zeros((n, n))
        cost_TZ_same = np.zeros((n, n, n))
        cost_TT_opp = np.zeros((n, n, n, n))
        cost_TT_same = np.zeros((n, n, n, n))
        kw = dict(cost_zz=cost_zz, cost_1q=cost_1q)
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                cost_T[p, q] = circuits.t_circuit_cost(p, q, **kw)
                cost_TZ_opp[p, q] = circuits.tz_opp_circuit_cost(p, q, **kw)
                for r in range(n):
                    if r != p and r != q:
                        cost_TZ_same[p, q, r] = circuits.tz_same_circuit_cost(
                            p, q, r, **kw
                        )
                    for s in range(n):
                        if s == r:
                            continue
                        cost_TT_opp[p, q, r, s] = circuits.tt_opp_circuit_cost(
                            p, q, r, s, **kw
                        )
                        if len({p, q, r, s}) == 4:
                            cost_TT_same[p, q, r, s] = (
                                circuits.tt_same_circuit_cost(p, q, r, s, **kw)
                            )
        tensors = {
            "Z": circuits.z_circuit_cost(**kw),
            "ZZ": circuits.zz_circuit_cost(**kw),
            "T": cost_T,
            "TZ_opp": cost_TZ_opp,
            "TZ_same": cost_TZ_same,
            "TT_opp": cost_TT_opp,
            "TT_same": cost_TT_same,
        }
        self._cost_tensors_cache[key] = tensors
        return tensors

    def jw_cost(
        self, perm: np.ndarray, *, cost_zz: float = 1.0, cost_1q: float = 0.0
    ) -> float:
        """Total weighted gate cost Σ_α |c_α| · gates_α(perm) under spatial-
        orbital permutation `perm` (perm[i] = orbital placed at JW position
        i within each spin sector).

        cost_zz and cost_1q select the gate-cost weights (see circuits.py).
        """
        n = self.num_orb
        pos = np.empty(n, dtype=int)
        pos[perm] = np.arange(n)
        t = self._circuit_cost_tensors(cost_zz=cost_zz, cost_1q=cost_1q)

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

    def optimize_jw_ordering(
        self, *, cost_zz: float = 1.0, cost_1q: float = 0.0
    ) -> np.ndarray:
        """Find a low-cost Jordan-Wigner ordering via spectral seeding +
        adjacent-swap refinement against `jw_cost`. Returns a permutation
        array π of length num_orb where π[i] is the spatial orbital placed
        at JW position i (within each spin sector).

        cost_zz and cost_1q are forwarded to `jw_cost` during refinement.
        """
        w = self._jw_pair_weights()
        n = self.num_orb

        # ── 1. Spectral seed (Fiedler vector of weighted Laplacian) ─────────
        degree = w.sum(axis=1)
        L = np.diag(degree) - w
        _, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
        perm = np.argsort(fiedler)

        # ── 2. Adjacent-swap refinement against actual jw_cost ──────────────
        kw = dict(cost_zz=cost_zz, cost_1q=cost_1q)
        improved = True
        while improved:
            improved = False
            current = self.jw_cost(perm, **kw)
            for i in range(n - 1):
                swapped = perm.copy()
                swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
                trial = self.jw_cost(swapped, **kw)
                if trial < current:
                    perm = swapped
                    current = trial
                    improved = True

        return perm

    def apply_jw_ordering(
        self, perm: np.ndarray = None, inplace: bool = False,
        *, cost_zz: float = 1.0, cost_1q: float = 0.0,
    ) -> "ElectronicStructureTZ | None":
        """Permute spatial orbital indices according to a JW ordering.

        :param perm: permutation array where perm[i] is the orbital placed at
                     position i. If None, calls optimize_jw_ordering() with
                     the given cost weights.
        :param inplace: if True, rebuild this instance in-place and return None;
                        if False, return a new ElectronicStructureTZ.
        """
        if perm is None:
            perm = self.optimize_jw_ordering(cost_zz=cost_zz, cost_1q=cost_1q)
        ix = np.ix_(perm, perm)
        ix4 = np.ix_(perm, perm, perm, perm)
        h1e_p = self.h1e[ix]
        h2e_p = self.h2e[ix4]
        if inplace:
            self.__init__(
                h0=self.h0, h1e=h1e_p, h2e=h2e_p, num_elec=self.num_elec
            )
            return None
        return ElectronicStructureTZ(
            h0=self.h0, h1e=h1e_p, h2e=h2e_p, num_elec=self.num_elec
        )

    def to_ElectronicStructureSZ(self) -> "ElectronicStructureSZ":
        """Build the SZ representation from this Hamiltonian's h0, h1e, h2e.

        ElectronicStructureSZ.__init__ derives all SZ coefficients directly
        from the integrals via the T_pq = S_pq - (I + Z_p Z_q)/2 substitution
        (see memory:project_sz_decomposition.md).
        """
        from champy.ElectronicStructureSZ import ElectronicStructureSZ

        return ElectronicStructureSZ(self.h0, self.h1e, self.h2e, self.num_elec)

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
                        _pauli_label({p + spin_offset: "Z", q + spin_offset: "Z"})
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
                        lxx_new, phase_xx = multiply_labels(lxx, z_label)
                        lyy_new, phase_yy = multiply_labels(lyy, z_label)
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
                                prod, phase = multiply_labels(l1, l2)
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
                            prod, phase = multiply_labels(l1, l2)
                            labels.append(prod)
                            weights.append(c / 4 * phase)

        return PauliHamiltonian.from_labels_and_weights(labels, weights)