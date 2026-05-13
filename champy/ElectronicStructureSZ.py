import functools

import numpy as np

from champy.ElectronicStructure import ElectronicStructure
from champy.ElectronicStructureTZ import ElectronicStructureTZ


class ElectronicStructureSZ(ElectronicStructure):
    """Electronic structure Hamiltonian in the S/Z operator basis.

    Built from the reflection operator
        S_pqx = T_pqx + (I + Z_px Z_qx) / 2
              = T_pqx + 1 - (n_px + n_qx - 2 n_px n_qx)
    and Pauli Z operators. Each S_pq and Z_p is hermitian, particle-number
    conserving, and squares to I (eigenvalues +-1). Products of S's and Z's
    on disjoint qubits inherit both properties, so every term in this
    representation is a number-preserving reflection.

    Operator types (grouped by structure and spin):
        constant
        Z_px (single Z)
        Z_px Z_qx (same-spin), Z_pu Z_qd (opposite-spin)
        Z_px Z_qx Z_rx (same-spin), Z_px Z_qx Z_ry (opposite-spin, x != y)
        Z_pu Z_qu Z_rd Z_sd (mixed spin), Z_px Z_qx Z_rx Z_sx (same-spin)
        S_pqx (single S)
        S_pqx Z_rx (same-spin, r not in {p,q}), S_pqx Z_ry (opposite-spin)
        S_pqx S_rsx (same-spin, {p,q} disjoint from {r,s}, p<r),
        S_pqu S_rsd (opposite-spin)
        S_pqx Z_rx Z_sx (same-spin, {p,q} disjoint from {r,s}),
        S_pqx Z_ry Z_sy (opposite-spin, x != y)
    """

    def __init__(self, h0: float, h1e: np.ndarray, h2e: np.ndarray, num_elec: int):
        super().__init__(h0=h0, h1e=h1e, h2e=h2e, num_elec=num_elec)

        coeffs = ElectronicStructureSZ._coefficients(h1e, h2e)
        tz_coeffs = ElectronicStructureTZ._coefficients(h1e, h2e)

        self._constant = float(
            h0
            + np.trace(h1e)
            - 0.25 * np.einsum("pqpq->", h2e)
            + 0.5 * np.einsum("ppqq->", h2e)
            - np.sum(tz_coeffs["T"])
            + np.sum(tz_coeffs["TT_opp"]) / 4
            + np.sum(tz_coeffs["TT_same"]) / 2
        )
        self.coeff_Z = np.array(coeffs["Z"])
        self.coeff_S = np.array(coeffs["S"])
        self.coeff_ZZ_same = np.array(coeffs["ZZ_same"])
        self.coeff_ZZ_opp = np.array(coeffs["ZZ_opp"])
        self.coeff_SZ_same = np.array(coeffs["SZ_same"])
        self.coeff_SZ_opp = np.array(coeffs["SZ_opp"])
        self.coeff_ZZZ_same = np.array(coeffs["ZZZ_same"])
        self.coeff_ZZZ_opp = np.array(coeffs["ZZZ_opp"])
        self.coeff_SS_same = np.array(coeffs["SS_same"])
        self.coeff_SS_opp = np.array(coeffs["SS_opp"])
        self.coeff_SZZ_same = np.array(coeffs["SZZ_same"])
        self.coeff_SZZ_opp = np.array(coeffs["SZZ_opp"])
        self.coeff_ZZZZ_same = np.array(coeffs["ZZZZ_same"])
        self.coeff_ZZZZ_opp = np.array(coeffs["ZZZZ_opp"])

    @staticmethod
    @functools.partial(__import__("jax").jit, static_argnums=())
    def _coefficients(h1e, h2e) -> dict:
        """Build the SZ operator-coefficient tensors from h1e, h2e.

        JIT-compiled and JAX-differentiable. Routes through
        ElectronicStructureTZ._coefficients (also jitted) and applies the
        TZ -> SZ substitution T_pq = S_pq - (I + Z_p Z_q)/2; see
        memory:project_sz_decomposition.md for the per-term derivation.
        """
        import jax.numpy as jnp
        tz = ElectronicStructureTZ._coefficients(h1e, h2e)
        cT, cZ = tz["T"], tz["Z"]
        cZZs_tz, cZZo = tz["ZZ_same"], tz["ZZ_opp"]
        cTZo, cTZs = tz["TZ_opp"], tz["TZ_same"]
        cTTo, cTTs = tz["TT_opp"], tz["TT_same"]

        sum_TTo_pq = jnp.einsum("pqrs->pq", cTTo)
        sum_TTs_pq = jnp.einsum("pqrs->pq", cTTs)
        sum_TTs_rs = jnp.einsum("pqrs->rs", cTTs)

        coeff_S = cT - 0.5 * sum_TTo_pq - 0.5 * (sum_TTs_pq + sum_TTs_rs)
        coeff_Z = (
            cZ
            - 0.5 * jnp.einsum("pqr->r", cTZo)
            - 0.5 * jnp.einsum("pqr->r", cTZs)
        )
        coeff_ZZ_same = (
            cZZs_tz
            - cT / 2
            + 0.25 * sum_TTo_pq
            + 0.25 * (sum_TTs_pq + sum_TTs_rs)
        )

        return {
            "Z": coeff_Z,
            "S": coeff_S,
            "ZZ_same": coeff_ZZ_same,
            "ZZ_opp": cZZo,
            "SZ_same": cTZs,
            "SZ_opp": cTZo,
            "ZZZ_same": -cTZs / 2,
            "ZZZ_opp": -cTZo / 2,
            "SS_same": cTTs,
            "SS_opp": cTTo,
            "SZZ_same": -(cTTs + cTTs.transpose(2, 3, 0, 1)) / 2,
            "SZZ_opp": -cTTo / 2,
            "ZZZZ_same": cTTs / 4,
            "ZZZZ_opp": cTTo / 4,
        }

    @property
    def constant(self) -> float:
        return self._constant

    def one_norm(self) -> float:
        """Sum of |coefficient| over all terms, with spin multiplicity.

        Each S, Z, and product term is hermitian and squares to I (S_pq has
        eigenvalues {+1,+1,+1,-1}; products on disjoint qubits inherit
        norm 1), so |c| is the operator-1-norm contribution per stored entry.

        Multiplicity rules:
        - same-spin terms (Z, S, ZZ_same, SZ_same, SS_same, ZZZ_same, ZZZZ_same,
          SZZ_same): coefficient applies to both alpha and beta -> ×2.
        - opp-spin terms with one operator type per index half (ZZ_opp, SS_opp,
          ZZZZ_opp): one coefficient = one operator (spin-flipped lives at the
          swapped index) -> ×1.
        - opp-spin terms with mixed operator types per index half (SZ_opp,
          ZZZ_opp, SZZ_opp): one coefficient covers both spin orderings at the
          same index -> ×2.
        """
        c = {
            "Z": self.coeff_Z,
            "S": self.coeff_S,
            "ZZ_same": self.coeff_ZZ_same,
            "ZZ_opp": self.coeff_ZZ_opp,
            "SZ_same": self.coeff_SZ_same,
            "SZ_opp": self.coeff_SZ_opp,
            "ZZZ_same": self.coeff_ZZZ_same,
            "ZZZ_opp": self.coeff_ZZZ_opp,
            "SS_same": self.coeff_SS_same,
            "SS_opp": self.coeff_SS_opp,
            "SZZ_same": self.coeff_SZZ_same,
            "SZZ_opp": self.coeff_SZZ_opp,
            "ZZZZ_same": self.coeff_ZZZZ_same,
            "ZZZZ_opp": self.coeff_ZZZZ_opp,
        }
        return float(ElectronicStructureSZ._one_norm_from_coeffs(np, c))

    @staticmethod
    def _one_norm(h1e, h2e):
        """JAX-differentiable 1-norm from integrals, for use in optimizers."""
        import jax.numpy as jnp

        c = ElectronicStructureSZ._coefficients(h1e, h2e)
        return ElectronicStructureSZ._one_norm_from_coeffs(jnp, c)

    @staticmethod
    def _one_norm_from_coeffs(xp, c):
        """Multiplicity-weighted sum of |c| over the 14 SZ term groups.

        See one_norm for the per-group multiplicity rules.
        """
        return (
            xp.sum(xp.abs(c["Z"])) * 2
            + xp.sum(xp.abs(c["S"])) * 2
            + xp.sum(xp.abs(c["ZZ_same"])) * 2
            + xp.sum(xp.abs(c["ZZ_opp"]))
            + xp.sum(xp.abs(c["SZ_same"])) * 2
            + xp.sum(xp.abs(c["SZ_opp"])) * 2
            + xp.sum(xp.abs(c["ZZZ_same"])) * 2
            + xp.sum(xp.abs(c["ZZZ_opp"])) * 2
            + xp.sum(xp.abs(c["SS_same"])) * 2
            + xp.sum(xp.abs(c["SS_opp"]))
            + xp.sum(xp.abs(c["SZZ_same"])) * 2
            + xp.sum(xp.abs(c["SZZ_opp"])) * 2
            + xp.sum(xp.abs(c["ZZZZ_same"])) * 2
            + xp.sum(xp.abs(c["ZZZZ_opp"]))
        )
