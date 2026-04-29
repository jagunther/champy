import numpy as np
from champy.PauliHamiltonian import PauliHamiltonian


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
    def _coefficients(h1e: np.ndarray, h2e: np.ndarray) -> dict:
        n = h1e.shape[0]
        mask_pq = np.triu(np.ones((n, n), dtype=bool), k=1)
        mask_rs = mask_pq
        coulomb = np.einsum("ppqq->pq", h2e)
        exchange = np.einsum("pqpq->pq", h2e)
        h_pqrr = np.einsum("pqrr->pqr", h2e)

        # Z_px: 1/2 (1/2 sum_q h_pqpq - h_pp - sum_q h_ppqq) per p
        coeff_Z = 0.5 * (
            0.5 * np.einsum("pqpq->p", h2e)
            - np.diag(h1e)
            - np.einsum("ppqq->p", h2e)
        )

        # T_pqx: (h_pq - 1/2 sum_r h_prrq + sum_r h_pqrr) per p<q
        coeff_T = np.triu(
            h1e - 0.5 * np.einsum("prrq->pq", h2e) + np.einsum("pqrr->pq", h2e),
            k=1,
        )

        # Z_px Z_qx (same-spin): 1/4 (h_ppqq - h_pqpq) per p<q
        coeff_ZZ_same = np.triu(0.25 * (coulomb - exchange), k=1)

        # Z_pu Z_qd (opposite-spin): 1/4 h_ppqq per (p, q)
        coeff_ZZ_opp = 0.25 * coulomb

        # TZ opposite-spin: -1/2 h_pqrr, all r
        coeff_TZ_opp = -0.5 * h_pqrr * mask_pq[:, :, None]

        # TZ same-spin: 1/2 (h_prrq - h_pqrr), r not in {p,q}
        h_prrq = np.einsum("prrq->pqr", h2e)
        coeff_TZ_same = 0.5 * (h_prrq - h_pqrr) * mask_pq[:, :, None]
        mask_r = np.ones((n, n, n), dtype=bool)
        for p in range(n):
            mask_r[p, :, p] = False
            mask_r[:, p, p] = False
        coeff_TZ_same = coeff_TZ_same * mask_r

        # TT masks
        mask_4d = mask_pq[:, :, None, None] & mask_rs[None, None, :, :]

        # same-spin: {p,q} ∩ {r,s} = ∅, canonical order p < r
        mask_4d_same = mask_4d.copy()
        for p in range(n):
            for q in range(p + 1, n):
                for r in range(n):
                    for s in range(r + 1, n):
                        if r <= p or len({p, q} & {r, s}) > 0:
                            mask_4d_same[p, q, r, s] = False
        coeff_TT_same = h2e * mask_4d_same

        # opposite-spin: all (p<q, r<s)
        coeff_TT_opp = h2e * mask_4d

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
        d>=2: Bell basis + parity tree (2d+1 entangling gates).
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
        s += _qasm_bell(qp, qq)
        s += f"cx q[{m}],q[{qp}];\n"
        s += f"rz({-angle}) q[{qp}];\n"
        s += f"rzz({angle}) q[{qp}],q[{qq}];\n"
        s += f"cx q[{m}],q[{qp}];\n"
        s += _qasm_bell_inv(qp, qq)
        s += _qasm_parity_tree_inv(inter)
        return s

    @staticmethod
    def tz_opp_circuit(
        p: int, q: int, x: int, r: int, angle: float, n: int, offset: int = 0
    ) -> str:
        """QASM circuit for exp(i * angle * T_pqx Z_ry) with y != x.

        Cost: 2d+3 entangling gates (d = |q-p|, d >= 2).
        """
        _q = ElectronicStructureTZ._qubit
        qp = _q(p, x, n, offset)
        qq = _q(q, x, n, offset)
        qr = _q(r, 1 - x, n, offset)
        d = abs(qq - qp)
        sign = 1 if qq > qp else -1
        inter = [qp + sign * i for i in range(1, d)]
        m = inter[0]

        s = ""
        s += _qasm_parity_tree(inter)
        s += _qasm_bell(qp, qq)
        s += f"cx q[{m}],q[{qp}];\n"
        s += f"cx q[{qr}],q[{qp}];\n"
        s += f"rz({-angle}) q[{qp}];\n"
        s += f"rzz({angle}) q[{qp}],q[{qq}];\n"
        s += f"cx q[{qr}],q[{qp}];\n"
        s += f"cx q[{m}],q[{qp}];\n"
        s += _qasm_bell_inv(qp, qq)
        s += _qasm_parity_tree_inv(inter)
        return s

    @staticmethod
    def tz_same_circuit(
        p: int, q: int, r: int, x: int, angle: float, n: int, offset: int = 0
    ) -> str:
        """QASM circuit for exp(i * angle * T_pqx Z_rx) with r not in {p,q}.

        Cases:
        - r outside [p,q]: same as opposite-spin case, 2d+3 entangling gates.
        - r inside (p,q) with d=2: Givens gate (2 entangling gates).
        - r inside (p,q) with d>=3: skip r in parity tree, 2(d-1) entangling gates.
        """
        _q = ElectronicStructureTZ._qubit
        qp = _q(p, x, n, offset)
        qq = _q(q, x, n, offset)
        qr = _q(r, x, n, offset)
        d = abs(qq - qp)
        sign = 1 if qq > qp else -1
        r_inside = (min(qp, qq) < qr < max(qp, qq))



        if not r_inside:
            inter = [qp + sign * i for i in range(1, d)]
            m = inter[0]
            s = ""
            s += _qasm_parity_tree(inter)
            s += _qasm_bell(qp, qq)
            s += f"cx q[{m}],q[{qp}];\n"
            s += f"cx q[{qr}],q[{qp}];\n"
            s += f"rz({-angle}) q[{qp}];\n"
            s += f"rzz({angle}) q[{qp}],q[{qq}];\n"
            s += f"cx q[{qr}],q[{qp}];\n"
            s += f"cx q[{m}],q[{qp}];\n"
            s += _qasm_bell_inv(qp, qq)
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
        s += _qasm_bell(qp, qq)
        s += f"cx q[{m}],q[{qp}];\n"
        s += f"rz({-angle}) q[{qp}];\n"
        s += f"rzz({angle}) q[{qp}],q[{qq}];\n"
        s += f"cx q[{m}],q[{qp}];\n"
        s += _qasm_bell_inv(qp, qq)
        s += _qasm_parity_tree_inv(inter)
        return s

    @staticmethod
    def tt_opp_circuit(
        p: int, q: int, r: int, s_orb: int, angle: float, n: int, offset: int = 0
    ) -> str:
        """QASM circuit for exp(i * angle * T_pqu T_rsd).

        Diagonalizes both T operators in parallel, then performs 4 diagonal
        ZZ rotations by accumulating parities onto qubit p.
        Cost: 2(d1 + d2) + 6 entangling gates (d1=|q-p|, d2=|s-r|, both >= 2).
        """
        _q = ElectronicStructureTZ._qubit
        qp = _q(p, 0, n, offset)
        qq = _q(q, 0, n, offset)
        qr = _q(r, 1, n, offset)
        qs = _q(s_orb, 1, n, offset)
        d1 = abs(qq - qp)
        d2 = abs(qs - qr)
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
        Cost: 2(d1 + d2) + 6 entangling gates (d1=|q-p|, d2=|s-r|).
        """
        _q = ElectronicStructureTZ._qubit
        qp = _q(p, x, n, offset)
        qq = _q(q, x, n, offset)
        qr = _q(r, x, n, offset)
        qs = _q(s_orb, x, n, offset)
        d1 = abs(qq - qp)
        d2 = abs(qs - qr)
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
