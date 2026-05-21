"""QASM circuit synthesis and weighted gate-cost counts for the T/Z and S/Z term families.

Each function emits exp(i * angle * P) for one operator pattern P appearing
in the TZ or SZ Hamiltonian decomposition (see ElectronicStructureTZ and
ElectronicStructureSZ for definitions). The matching *_circuit_cost functions
return a weighted cost without materialising the circuit, used by jw_cost and
other resource estimates.

Cost model
----------
Each cost function accepts keyword-only ``cost_zz`` (weight per 2-qubit
entangling gate, default 1.0) and ``cost_1q`` (weight per 1-qubit rotation
"block", default 0.0) and returns ``n_zz * cost_zz + n_1q * cost_1q``.

- 2-qubit entangling gates: each ``cx`` and each ``rzz(...)`` macro call
  counts once. ``rzz`` is treated as atomic (not as its 2-CX + Rz expansion).
- 1-qubit rotation blocks: counted per qubit, with the rule that consecutive
  1q gates on the same qubit (with no intervening 2q gate touching that
  qubit) fuse into a single block. All 1q gates are treated alike —
  Cliffords (h, s, sdg) and rotations (rx, ry, rz at any angle) each
  contribute to whichever block they fall into.
- Convenience macros (``givens``, ``xxyy_zbasis``, ``xxyy_zbasis_inv``,
  ``xyswap``) are expanded to their bodies for counting.

With the default ``cost_zz=1, cost_1q=0`` the returned value equals the
classic entangling-gate count used previously.

Orbitals are indexed (p, x) with p ∈ [0, n) the spatial-orbital index and
x ∈ {0, 1} the spin sector; qubit_index packs them into a JW qubit number.
"""

import functools

import numpy as np

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


def qubit_index(p: int, x: int, n: int, offset: int = 0) -> int:
    """JW qubit index for spatial orbital p, spin x, with optional offset."""
    return p - 1 + x * n + offset


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


def _qasm_xy_swap(qubit: int) -> str:
    """QASM for X<->Y swap gate: Rx(-pi/2) H Rx(pi/2). Self-inverse."""
    return f"xyswap q[{qubit}];\n"


# ── Pure-Z products (basis-agnostic) ────────────────────────────────────────
#
# Trotter steps for products of Z operators. Used by both TZ- and SZ-basis
# Hamiltonians. Each circuit's matching circuit_cost returns its 2-qubit-gate
# count.


def z_circuit(p: int, x: int, angle: float, n: int, offset: int = 0) -> str:
    """QASM circuit for exp(i * angle * Z_px)."""
    q = qubit_index(p, x, n, offset)
    return f"rz({-2 * angle}) q[{q}];\n"


def z_circuit_cost(*, cost_zz: float = 1.0, cost_1q: float = 0.0) -> float:
    """Weighted cost for z_circuit: 1 Rz rotation."""
    return cost_1q


def zz_circuit(
    p: int, x_p: int, q: int, x_q: int, angle: float, n: int, offset: int = 0
) -> str:
    """QASM circuit for exp(i * angle * Z_px Z_qy)."""
    q0 = qubit_index(p, x_p, n, offset)
    q1 = qubit_index(q, x_q, n, offset)
    return f"rzz({-2 * angle}) q[{q0}],q[{q1}];\n"


def zz_circuit_cost(*, cost_zz: float = 1.0, cost_1q: float = 0.0) -> float:
    """Weighted cost for zz_circuit: 1 RZZ rotation."""
    return cost_zz


def _z_string_circuit(qubits: list[int], angle: float) -> str:
    """QASM for exp(i*angle * Z_q0 Z_q1 ... Z_qk-1).

    k=1: single-qubit Rz. k>=2: split the qubits into two non-empty
    groups, parity-tree each group onto its first qubit, then do one RZZ
    between the two accumulators, then uncompute the parity trees. This
    gives 2k-3 entangling gates: 1, 3, 5 for k=2, 3, 4.
    """
    if len(qubits) == 1:
        return f"rz({-2 * angle}) q[{qubits[0]}];\n"
    mid = (len(qubits) + 1) // 2
    left, right = qubits[:mid], qubits[mid:]
    s = ""
    for k in left[1:]:
        s += f"cx q[{k}],q[{left[0]}];\n"
    for k in right[1:]:
        s += f"cx q[{k}],q[{right[0]}];\n"
    s += f"rzz({-2 * angle}) q[{left[0]}],q[{right[0]}];\n"
    for k in reversed(right[1:]):
        s += f"cx q[{k}],q[{right[0]}];\n"
    for k in reversed(left[1:]):
        s += f"cx q[{k}],q[{left[0]}];\n"
    return s


def zzz_same_circuit(
    p: int, q: int, r: int, x: int, angle: float, n: int, offset: int = 0
) -> str:
    """QASM for exp(i*angle * Z_px Z_qx Z_rx). 4 entangling gates."""
    qubits = [qubit_index(o, x, n, offset) for o in (p, q, r)]
    return _z_string_circuit(qubits, angle)


def zzz_opp_circuit(
    p: int, q: int, r: int, x: int, angle: float, n: int, offset: int = 0
) -> str:
    """QASM for exp(i*angle * Z_px Z_qx Z_r^(1-x)). 4 entangling gates.

    (p,q) is the same-spin pair, r is on the opposite spin.
    """
    qubits = [
        qubit_index(p, x, n, offset),
        qubit_index(q, x, n, offset),
        qubit_index(r, 1 - x, n, offset),
    ]
    return _z_string_circuit(qubits, angle)


def zzz_circuit_cost(*, cost_zz: float = 1.0, cost_1q: float = 0.0) -> float:
    """Weighted cost for zzz_*_circuit: 2 CX + 1 RZZ from _z_string_circuit with 3 qubits."""
    return 3 * cost_zz


def zzzz_same_circuit(
    p: int,
    q: int,
    r: int,
    s: int,
    x: int,
    angle: float,
    n: int,
    offset: int = 0,
) -> str:
    """QASM for exp(i*angle * Z_px Z_qx Z_rx Z_sx). 6 entangling gates."""
    qubits = [qubit_index(o, x, n, offset) for o in (p, q, r, s)]
    return _z_string_circuit(qubits, angle)


def zzzz_opp_circuit(
    p: int,
    q: int,
    r: int,
    s: int,
    angle: float,
    n: int,
    offset: int = 0,
) -> str:
    """QASM for exp(i*angle * Z_p^α Z_q^α Z_r^β Z_s^β). 6 entangling gates.

    Convention matches coeff_ZZZZ_opp storage: alpha pair = (p,q),
    beta pair = (r,s).
    """
    qubits = [
        qubit_index(p, 0, n, offset),
        qubit_index(q, 0, n, offset),
        qubit_index(r, 1, n, offset),
        qubit_index(s, 1, n, offset),
    ]
    return _z_string_circuit(qubits, angle)


def zzzz_circuit_cost(*, cost_zz: float = 1.0, cost_1q: float = 0.0) -> float:
    """Weighted cost for zzzz_*_circuit: 4 CX + 1 RZZ from _z_string_circuit with 4 qubits."""
    return 5 * cost_zz


# ── T-family (TZ-basis) ─────────────────────────────────────────────────────
#
# Trotter steps for products involving the hopping operator
# T_pq = (X_p Z_{p+1}...Z_{q-1} X_q + Y_p Z_{p+1}...Z_{q-1} Y_q) / 2.
# The same-spin TT family has three topological cases (non-overlap,
# interleaved, nested) with different gate counts, dispatched by
# tt_same_circuit_cost.


# top-level circuit dispatcher for all TZ terms
def tz_term_circuit(term, angle: float, n: int, offset: int) -> str:
    """Emit the QASM body for exp(i · angle · O_α) for the given term.
    Term is tuple as given by ElectronicStructureTZ.enumerate_terms()."""
    kind, indices, x, _, _ = term
    if kind == "Z":
        (p,) = indices
        return z_circuit(p + 1, x, angle, n, offset)
    if kind == "T":
        p, q = indices
        return t_circuit(p + 1, q + 1, x, angle, n, offset)
    if kind == "ZZ_same":
        p, q = indices
        return zz_circuit(p + 1, x, q + 1, x, angle, n, offset)
    if kind == "ZZ_opp":
        p, q = indices
        return zz_circuit(p + 1, 0, q + 1, 1, angle, n, offset)
    if kind == "TZ_opp":
        p, q, r = indices
        return tz_opp_circuit(p + 1, q + 1, x, r + 1, angle, n, offset)
    if kind == "TZ_same":
        p, q, r = indices
        return tz_same_circuit(p + 1, q + 1, r + 1, x, angle, n, offset)
    if kind == "TT_opp":
        p, q, r, s = indices
        return tt_opp_circuit(p + 1, q + 1, r + 1, s + 1, angle, n, offset)
    if kind == "TT_same":
        # mask_tt_same guarantees p < q, r < s, p < r, {p,q} ∩ {r,s} = ∅.
        # With p < r, the three topologies reduce to: non-overlap (q < r),
        # interleaved (r < q < s), nested (r < s < q).
        p, q, r, s = indices
        if q < r:
            return tt_same_nonoverlap_circuit(
                p + 1, q + 1, r + 1, s + 1, x, angle, n, offset
            )
        if q < s:
            return tt_same_interleaved_circuit(
                p + 1, q + 1, r + 1, s + 1, x, angle, n, offset
            )
        return tt_same_nested_circuit(p + 1, q + 1, r + 1, s + 1, x, angle, n, offset)
    raise ValueError(f"Unknown TZ term kind '{kind}'")


def t_circuit(p: int, q: int, x: int, angle: float, n: int, offset: int = 0) -> str:
    """QASM circuit for exp(i * angle * T_pqx), where T_pq = (XX+YY)/2.

    d=1: Givens gate (2 entangling gates).
    d>=2: parity tree + z_xxyy (2d entangling gates).
    Requires QASM_GATE_DEFS to be prepended to the program.
    """
    qp = qubit_index(p, x, n, offset)
    qq = qubit_index(q, x, n, offset)
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


def t_circuit_cost(
    p: int, q: int, *, cost_zz: float = 1.0, cost_1q: float = 0.0
) -> float:
    """Weighted cost for t_circuit.

    d=1: Givens (2 CX + 6 fused 1q-blocks).
    d>=2: parity tree + 2 xxyy_zbasis + 2 RZZ (2d CX-equivalent, 8 1q-blocks
    from the two xxyy macros).
    """
    d = abs(q - p)
    if d == 1:
        return 2 * cost_zz + 6 * cost_1q
    return 2 * d * cost_zz + 8 * cost_1q


def tz_opp_circuit(
    p: int, q: int, x: int, r: int, angle: float, n: int, offset: int = 0
) -> str:
    """QASM circuit for exp(i * angle * T_pqx Z_ry) with y != x.

    d=1: xxyy_zbasis + 2 RZZ on (r, p) and (r, q) (4 entangling gates).
    d>=2: parity tree + Z_r XOR + xxyy_zbasis + 2 RZZ + undo (2d+2 entangling gates).
    """
    qp = qubit_index(p, x, n, offset)
    qq = qubit_index(q, x, n, offset)
    qr = qubit_index(r, 1 - x, n, offset)
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


def tz_opp_circuit_cost(
    p: int, q: int, *, cost_zz: float = 1.0, cost_1q: float = 0.0
) -> float:
    """Weighted cost for tz_opp_circuit (independent of r).

    d=1: 4 cost_zz (xxyy + 2 RZZ + xxyy_inv) + 8 cost_1q from 2 xxyy macros.
    d>=2: (2d+2) cost_zz + 8 cost_1q.
    """
    d = abs(q - p)
    if d == 1:
        return 4 * cost_zz + 8 * cost_1q
    return (2 * d + 2) * cost_zz + 8 * cost_1q


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
    qp = qubit_index(p, x, n, offset)
    qq = qubit_index(q, x, n, offset)
    qr = qubit_index(r, x, n, offset)
    d = abs(qq - qp)
    sign = 1 if qq > qp else -1
    r_inside = min(qp, qq) < qr < max(qp, qq)

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


def tz_same_circuit_cost(
    p: int, q: int, r: int, *, cost_zz: float = 1.0, cost_1q: float = 0.0
) -> float:
    """Weighted cost for tz_same_circuit.

    r outside [p,q], d=1: 4 cost_zz + 8 cost_1q (same as tz_opp d=1).
    r outside, d>=2: (2d+2) cost_zz + 8 cost_1q.
    r inside, d=2: Givens, 2 cost_zz + 6 cost_1q.
    r inside, d>=3: 2(d-1) cost_zz + 8 cost_1q.
    """
    d = abs(q - p)
    r_inside = min(p, q) < r < max(p, q)
    if not r_inside:
        if d == 1:
            return 4 * cost_zz + 8 * cost_1q
        return (2 * d + 2) * cost_zz + 8 * cost_1q
    if d == 2:
        return 2 * cost_zz + 6 * cost_1q
    return 2 * (d - 1) * cost_zz + 8 * cost_1q


def _t_zstring_circuit(qp: int, qq: int, eff_qubits: list[int], angle: float) -> str:
    """QASM for exp(i*angle * T_{qp,qq} · Π_{k∈eff} Z_k) given absolute qubit indices.

    Uses the same xxyy_zbasis pattern as t_circuit / tz_*_circuit:
    accumulate parity of `eff_qubits` onto eff_qubits[0], change to Z basis on
    (qp,qq), apply 2 RZZ to that accumulator, undo. eff_qubits must be disjoint
    from {qp, qq}. If empty, the (XX+YY)/2 piece becomes -(Z_p+Z_q)/2 and we
    apply two single-qubit Rz instead. Cost: 2 if eff is empty, else 2*len+2.
    """
    s = ""
    if not eff_qubits:
        s += f"xxyy_zbasis q[{qp}],q[{qq}];\n"
        s += f"rz({angle}) q[{qp}];\n"
        s += f"rz({angle}) q[{qq}];\n"
        s += f"xxyy_zbasis_inv q[{qp}],q[{qq}];\n"
        return s
    m = eff_qubits[0]
    for k in eff_qubits[1:]:
        s += f"cx q[{k}],q[{m}];\n"
    s += f"xxyy_zbasis q[{qp}],q[{qq}];\n"
    s += f"rzz({angle}) q[{m}],q[{qp}];\n"
    s += f"rzz({angle}) q[{m}],q[{qq}];\n"
    s += f"xxyy_zbasis_inv q[{qp}],q[{qq}];\n"
    for k in reversed(eff_qubits[1:]):
        s += f"cx q[{k}],q[{m}];\n"
    return s


def tzz_same_circuit(
    p: int,
    q: int,
    r: int,
    s_orb: int,
    x: int,
    angle: float,
    n: int,
    offset: int = 0,
) -> str:
    """QASM for exp(i*angle * T_pqx Z_rx Z_sx) with {p,q} ∩ {r,s} = ∅.

    Effective Z-string: symmetric difference of (intermediates of (p,q))
    and {q_r, q_s}. r or s inside the (p,q) interval cancels with the
    JW string.
    """
    qp = qubit_index(p, x, n, offset)
    qq = qubit_index(q, x, n, offset)
    qr = qubit_index(r, x, n, offset)
    qs = qubit_index(s_orb, x, n, offset)
    d = abs(qq - qp)
    sign = 1 if qq > qp else -1
    inter_set = set(qp + sign * i for i in range(1, d))
    eff = sorted(inter_set ^ {qr, qs})
    return _t_zstring_circuit(qp, qq, eff, angle)


def tzz_same_circuit_cost(
    p: int,
    q: int,
    r: int,
    s_orb: int,
    *,
    cost_zz: float = 1.0,
    cost_1q: float = 0.0,
) -> float:
    """Weighted cost for tzz_same_circuit. Eff size = d+1 - 2·(# of {r,s} inside (p,q)).

    eff_size == 0: xxyy + 2 Rz + xxyy_inv → 2 cost_zz + 6 cost_1q (the 2 Rz
    fuse with the trailing/leading 1q gates of the surrounding xxyy macros).
    eff_size > 0: parity tree + 2 xxyy + 2 RZZ + uncompute →
    (2*eff_size + 2) cost_zz + 8 cost_1q.
    """
    d = abs(q - p)
    lo, hi = min(p, q), max(p, q)
    inside = (1 if lo < r < hi else 0) + (1 if lo < s_orb < hi else 0)
    eff_size = d + 1 - 2 * inside
    if eff_size == 0:
        return 2 * cost_zz + 6 * cost_1q
    return (2 * eff_size + 2) * cost_zz + 8 * cost_1q


def tzz_opp_circuit(
    p: int,
    q: int,
    r: int,
    s_orb: int,
    x: int,
    angle: float,
    n: int,
    offset: int = 0,
) -> str:
    """QASM for exp(i*angle * T_pqx Z_r^(1-x) Z_s^(1-x)).

    (p,q) on spin x; (r,s) on the opposite spin (no qubit overlap with
    intermediates), so the effective string is the union of intermediates
    and {q_r, q_s}.
    """
    qp = qubit_index(p, x, n, offset)
    qq = qubit_index(q, x, n, offset)
    qr = qubit_index(r, 1 - x, n, offset)
    qs = qubit_index(s_orb, 1 - x, n, offset)
    d = abs(qq - qp)
    sign = 1 if qq > qp else -1
    inter_set = set(qp + sign * i for i in range(1, d))
    eff = sorted(inter_set | {qr, qs})
    return _t_zstring_circuit(qp, qq, eff, angle)


def tzz_opp_circuit_cost(
    p: int, q: int, *, cost_zz: float = 1.0, cost_1q: float = 0.0
) -> float:
    """Weighted cost for tzz_opp_circuit. Eff size = (d-1) + 2 = d+1 (always nonempty).
    (2*(d+1) + 2) cost_zz + 8 cost_1q from 2 xxyy macros.
    """
    d = abs(q - p)
    return (2 * (d + 1) + 2) * cost_zz + 8 * cost_1q


def tt_opp_circuit(
    p: int, q: int, r: int, s_orb: int, angle: float, n: int, offset: int = 0
) -> str:
    """QASM circuit for exp(i * angle * T_pqu T_rsd).

    Uses xxyy_zbasis to map (XX+YY)/2 -> -(Z_p+Z_q)/2 on each pair, then
    absorbs the Z-string parity (if any) onto p and q via CNOTs, and
    performs 4 RZZ rotations on (p,r), (p,s), (q,r), (q,s).
    Cost: 8 if d1=d2=1, otherwise 2(d1+d2)+6.
    """
    qp = qubit_index(p, 0, n, offset)
    qq = qubit_index(q, 0, n, offset)
    qr = qubit_index(r, 1, n, offset)
    qs = qubit_index(s_orb, 1, n, offset)
    d1 = abs(qq - qp)
    d2 = abs(qs - qr)
    sign1 = 1 if qq > qp else -1
    sign2 = 1 if qs > qr else -1
    inter1 = [qp + sign1 * i for i in range(1, d1)]
    inter2 = [qr + sign2 * i for i in range(1, d2)]
    # m: combined Z-string accumulator. None if both T's have no Z-string.
    if inter1 and inter2:
        m = inter1[0]
        combine = (inter2[0], inter1[0])
    elif inter1:
        m = inter1[0]
        combine = None
    elif inter2:
        m = inter2[0]
        combine = None
    else:
        m = None
        combine = None

    s = ""
    s += _qasm_parity_tree(inter1)
    s += _qasm_parity_tree(inter2)
    if combine is not None:
        s += f"cx q[{combine[0]}],q[{combine[1]}];\n"
    s += f"xxyy_zbasis q[{qp}],q[{qq}];\n"
    s += f"xxyy_zbasis q[{qr}],q[{qs}];\n"
    if m is not None:
        s += f"cx q[{m}],q[{qp}];\n"
        s += f"cx q[{m}],q[{qq}];\n"
    s += f"rzz({-angle/2}) q[{qp}],q[{qr}];\n"
    s += f"rzz({-angle/2}) q[{qp}],q[{qs}];\n"
    s += f"rzz({-angle/2}) q[{qq}],q[{qr}];\n"
    s += f"rzz({-angle/2}) q[{qq}],q[{qs}];\n"
    if m is not None:
        s += f"cx q[{m}],q[{qq}];\n"
        s += f"cx q[{m}],q[{qp}];\n"
    s += f"xxyy_zbasis_inv q[{qr}],q[{qs}];\n"
    s += f"xxyy_zbasis_inv q[{qp}],q[{qq}];\n"
    if combine is not None:
        s += f"cx q[{combine[0]}],q[{combine[1]}];\n"
    s += _qasm_parity_tree_inv(inter2)
    s += _qasm_parity_tree_inv(inter1)
    return s


def tt_opp_circuit_cost(
    p: int,
    q: int,
    r: int,
    s: int,
    *,
    cost_zz: float = 1.0,
    cost_1q: float = 0.0,
) -> float:
    """Weighted cost for tt_opp_circuit. 16 cost_1q always (4 xxyy macros total).

    d1=d2=1: 2 xxyy + 4 RZZ + 2 xxyy_inv = 8 cost_zz.
    Otherwise: 2(d1+d2)+6 cost_zz (parity trees + combine + xxyy + m-absorption + 4 RZZ).
    """
    d1 = abs(q - p)
    d2 = abs(s - r)
    if d1 == 1 and d2 == 1:
        return 8 * cost_zz + 16 * cost_1q
    return (2 * (d1 + d2) + 6) * cost_zz + 16 * cost_1q


def tt_same_nonoverlap_circuit(
    p: int,
    q: int,
    r: int,
    s_orb: int,
    x: int,
    angle: float,
    n: int,
    offset: int = 0,
) -> str:
    """QASM circuit for exp(i * angle * T_pqx T_rsx), non-overlapping case.

    Requires p<q<r<s (or r<s<p<q). Uses xxyy_zbasis on each pair, combines
    Z-string accumulators when both exist, absorbs onto p and q, then
    performs 4 RZZ rotations on (p,r), (p,s), (q,r), (q,s).
    Cost: 8 if d1=d2=1, otherwise 2(d1+d2)+6.
    """
    qp = qubit_index(p, x, n, offset)
    qq = qubit_index(q, x, n, offset)
    qr = qubit_index(r, x, n, offset)
    qs = qubit_index(s_orb, x, n, offset)
    d1 = abs(qq - qp)
    d2 = abs(qs - qr)
    sign1 = 1 if qq > qp else -1
    sign2 = 1 if qs > qr else -1
    inter1 = [qp + sign1 * i for i in range(1, d1)]
    inter2 = [qr + sign2 * i for i in range(1, d2)]
    if inter1 and inter2:
        m = inter1[0]
        combine = (inter2[0], inter1[0])
    elif inter1:
        m = inter1[0]
        combine = None
    elif inter2:
        m = inter2[0]
        combine = None
    else:
        m = None
        combine = None

    s = ""
    s += _qasm_parity_tree(inter1)
    s += _qasm_parity_tree(inter2)
    if combine is not None:
        s += f"cx q[{combine[0]}],q[{combine[1]}];\n"
    s += f"xxyy_zbasis q[{qp}],q[{qq}];\n"
    s += f"xxyy_zbasis q[{qr}],q[{qs}];\n"
    if m is not None:
        s += f"cx q[{m}],q[{qp}];\n"
        s += f"cx q[{m}],q[{qq}];\n"
    s += f"rzz({-angle/2}) q[{qp}],q[{qr}];\n"
    s += f"rzz({-angle/2}) q[{qp}],q[{qs}];\n"
    s += f"rzz({-angle/2}) q[{qq}],q[{qr}];\n"
    s += f"rzz({-angle/2}) q[{qq}],q[{qs}];\n"
    if m is not None:
        s += f"cx q[{m}],q[{qq}];\n"
        s += f"cx q[{m}],q[{qp}];\n"
    s += f"xxyy_zbasis_inv q[{qr}],q[{qs}];\n"
    s += f"xxyy_zbasis_inv q[{qp}],q[{qq}];\n"
    if combine is not None:
        s += f"cx q[{combine[0]}],q[{combine[1]}];\n"
    s += _qasm_parity_tree_inv(inter2)
    s += _qasm_parity_tree_inv(inter1)
    return s


def tt_same_interleaved_circuit(
    p: int,
    q: int,
    r: int,
    s_orb: int,
    x: int,
    angle: float,
    n: int,
    offset: int = 0,
) -> str:
    """QASM circuit for exp(i * angle * T_pqx T_rsx), interleaved p<r<q<s or r<p<s<q.

    X<->Y swap on the two inner qubits (a,c) aligns the operator into
    matched XX/YY structure. Then xxyy_zbasis on (a,b) and (c,d), absorb
    Z-string accumulator onto qa,qb, and 4 RZZ rotations.
    Cost: 8 if no outer intermediates, otherwise 10 + extra(δ1) + extra(δ2).
    """
    qp = qubit_index(p, x, n, offset)
    qq = qubit_index(q, x, n, offset)
    qr = qubit_index(r, x, n, offset)
    qs = qubit_index(s_orb, x, n, offset)
    # Identify interleaved ordering: a < c < b < d
    if qp < qr:
        qa, qb, qc, qd = qp, qq, qr, qs
    else:
        qa, qb, qc, qd = qr, qs, qp, qq
    inter1 = list(range(qa + 1, qc))
    inter2 = list(range(qb + 1, qd))
    if inter1 and inter2:
        m = inter1[0]
        combine = (inter2[0], inter1[0])
    elif inter1:
        m = inter1[0]
        combine = None
    elif inter2:
        m = inter2[0]
        combine = None
    else:
        m = None
        combine = None

    s = ""
    s += _qasm_xy_swap(qa)
    s += _qasm_xy_swap(qc)
    s += _qasm_parity_tree(inter1)
    s += _qasm_parity_tree(inter2)
    if combine is not None:
        s += f"cx q[{combine[0]}],q[{combine[1]}];\n"
    s += f"xxyy_zbasis q[{qa}],q[{qb}];\n"
    s += f"xxyy_zbasis q[{qc}],q[{qd}];\n"
    if m is not None:
        s += f"cx q[{m}],q[{qa}];\n"
        s += f"cx q[{m}],q[{qb}];\n"
    # target ~ (Z_a - Z_b)(Z_d - Z_c)/4 = (+Z_aZ_d - Z_aZ_c - Z_bZ_d + Z_bZ_c)/4
    s += f"rzz({-angle/2}) q[{qa}],q[{qd}];\n"
    s += f"rzz({angle/2}) q[{qa}],q[{qc}];\n"
    s += f"rzz({angle/2}) q[{qb}],q[{qd}];\n"
    s += f"rzz({-angle/2}) q[{qb}],q[{qc}];\n"
    if m is not None:
        s += f"cx q[{m}],q[{qb}];\n"
        s += f"cx q[{m}],q[{qa}];\n"
    s += f"xxyy_zbasis_inv q[{qc}],q[{qd}];\n"
    s += f"xxyy_zbasis_inv q[{qa}],q[{qb}];\n"
    if combine is not None:
        s += f"cx q[{combine[0]}],q[{combine[1]}];\n"
    s += _qasm_parity_tree_inv(inter2)
    s += _qasm_parity_tree_inv(inter1)
    s += _qasm_xy_swap(qc)
    s += _qasm_xy_swap(qa)
    return s


def tt_same_nested_circuit(
    p: int,
    q: int,
    r: int,
    s_orb: int,
    x: int,
    angle: float,
    n: int,
    offset: int = 0,
) -> str:
    """QASM circuit for exp(i * angle * T_pqx T_rsx), nested case p<r<s<q or r<p<q<s.

    The outer pair's Z-string contains both inner qubits. After parity trees
    on outer intermediates (a+1..b-1) and (c+1..d-1), the Z-string accumulator
    gets absorbed onto the outer-pair qubits qa, qd. Then xxyy_zbasis on
    outer (a,d) and inner (b,c), and 4 RZZ rotations connecting outer to inner.
    Cost: 8 if no outer intermediates, otherwise 10 + extra(δ1) + extra(δ2).
    """
    qp = qubit_index(p, x, n, offset)
    qq = qubit_index(q, x, n, offset)
    qr = qubit_index(r, x, n, offset)
    qs = qubit_index(s_orb, x, n, offset)
    # Identify nesting: outer (a,d) contains inner (b,c), a<b<c<d
    if qp < qr:
        qa, qb, qc, qd = qp, qr, qs, qq
    else:
        qa, qb, qc, qd = qr, qp, qq, qs
    inter1 = list(range(qa + 1, qb))
    inter2 = list(range(qc + 1, qd))
    if inter1 and inter2:
        m = inter1[0]
        combine = (inter2[0], inter1[0])
    elif inter1:
        m = inter1[0]
        combine = None
    elif inter2:
        m = inter2[0]
        combine = None
    else:
        m = None
        combine = None

    s = ""
    s += _qasm_parity_tree(inter1)
    s += _qasm_parity_tree(inter2)
    if combine is not None:
        s += f"cx q[{combine[0]}],q[{combine[1]}];\n"
    s += f"xxyy_zbasis q[{qa}],q[{qd}];\n"
    s += f"xxyy_zbasis q[{qb}],q[{qc}];\n"
    if m is not None:
        s += f"cx q[{m}],q[{qa}];\n"
        s += f"cx q[{m}],q[{qd}];\n"
    # target ~ -(Z_a + Z_d)(Z_b + Z_c)/4 = -(Z_aZ_b + Z_aZ_c + Z_bZ_d + Z_cZ_d)/4
    s += f"rzz({angle/2}) q[{qa}],q[{qb}];\n"
    s += f"rzz({angle/2}) q[{qa}],q[{qc}];\n"
    s += f"rzz({angle/2}) q[{qb}],q[{qd}];\n"
    s += f"rzz({angle/2}) q[{qc}],q[{qd}];\n"
    if m is not None:
        s += f"cx q[{m}],q[{qd}];\n"
        s += f"cx q[{m}],q[{qa}];\n"
    s += f"xxyy_zbasis_inv q[{qb}],q[{qc}];\n"
    s += f"xxyy_zbasis_inv q[{qa}],q[{qd}];\n"
    if combine is not None:
        s += f"cx q[{combine[0]}],q[{combine[1]}];\n"
    s += _qasm_parity_tree_inv(inter2)
    s += _qasm_parity_tree_inv(inter1)
    return s


def tt_same_circuit_cost(
    p: int,
    q: int,
    r: int,
    s: int,
    *,
    cost_zz: float = 1.0,
    cost_1q: float = 0.0,
) -> float:
    """Weighted cost for same-spin TT (nonoverlap, interleaved, or nested).

    Topology only affects cost_zz; all three variants have 16 cost_1q from
    the four xxyy_zbasis macros (the 4 xy_swaps in the interleaved variant
    fuse into the adjacent xxyy 1q-runs and add no new blocks).

    Dispatches based on the qubit ordering of (p,q) and (r,s):
    - non-overlapping: 2(d1+d2)+6 cost_zz (or 8 if d1=d2=1).
    - interleaved or nested: 10 + extra(δ1) + extra(δ2) cost_zz (or 8 if δ1=δ2=1).
    Here extra(k) = 2k-2 if k>=2 else 0.
    """
    lo1, hi1 = min(p, q), max(p, q)
    lo2, hi2 = min(r, s), max(r, s)
    if lo1 > lo2:
        lo1, hi1, lo2, hi2 = lo2, hi2, lo1, hi1

    if hi1 < lo2:
        # nonoverlap
        d1, d2 = hi1 - lo1, hi2 - lo2
        n_zz = 8 if (d1 == 1 and d2 == 1) else 2 * (d1 + d2) + 6
    else:
        # interleaved (hi1 < hi2) or nested (hi1 >= hi2)
        if hi1 < hi2:
            delta1 = lo2 - lo1
            delta2 = hi2 - hi1
        else:
            delta1 = lo2 - lo1
            delta2 = hi1 - hi2
        if delta1 == 1 and delta2 == 1:
            n_zz = 8
        else:
            extra = lambda k: 2 * k - 2 if k >= 2 else 0
            n_zz = 10 + extra(delta1) + extra(delta2)
    return n_zz * cost_zz + 16 * cost_1q


# ── S-family (SZ-basis) ─────────────────────────────────────────────────────
#
# Trotter steps for products involving the reflection operator
# S_pq = T_pq + (I + Z_p Z_q)/2. All four Paulis in S_pq commute, so each
# S-containing factor decomposes into commuting T- and pure-Z pieces.
# Discarding the global phase from the I in S_pq:
#   exp(iθ · S_pq)         = exp(iθ · T_pq) · exp(iθ/2 · Z_p Z_q)
#   exp(iθ · S_pq Z_r)     = exp(iθ/2 · Z_r) · exp(iθ · T_pq Z_r)
#                            · exp(iθ/2 · Z_p Z_q Z_r)
#   exp(iθ · S_pq Z_r Z_s) = exp(iθ/2 · Z_r Z_s) · exp(iθ · T_pq Z_r Z_s)
#                            · exp(iθ/2 · Z_p Z_q Z_r Z_s)


def s_circuit(p: int, q: int, x: int, angle: float, n: int, offset: int = 0) -> str:
    """QASM for exp(i*angle * S_pqx). Cost: t_circuit_cost(p,q) + 1."""
    s = ""
    s += t_circuit(p, q, x, angle, n, offset)
    s += zz_circuit(p, x, q, x, angle / 2, n, offset)
    return s


def s_circuit_cost(
    p: int, q: int, *, cost_zz: float = 1.0, cost_1q: float = 0.0
) -> float:
    """Weighted cost for s_circuit: t_circuit + zz_circuit(p,x,q,x,angle/2)."""
    return t_circuit_cost(p, q, cost_zz=cost_zz, cost_1q=cost_1q) + zz_circuit_cost(
        cost_zz=cost_zz, cost_1q=cost_1q
    )


def sz_same_circuit(
    p: int,
    q: int,
    r: int,
    x: int,
    angle: float,
    n: int,
    offset: int = 0,
) -> str:
    """QASM for exp(i*angle * S_pqx Z_rx) with r not in {p,q}."""
    s = ""
    s += z_circuit(r, x, angle / 2, n, offset)
    s += tz_same_circuit(p, q, r, x, angle, n, offset)
    s += zzz_same_circuit(p, q, r, x, angle / 2, n, offset)
    return s


def sz_same_circuit_cost(
    p: int, q: int, r: int, *, cost_zz: float = 1.0, cost_1q: float = 0.0
) -> float:
    """Weighted cost for sz_same_circuit: z_circuit + tz_same_circuit + zzz_same_circuit."""
    return (
        z_circuit_cost(cost_zz=cost_zz, cost_1q=cost_1q)
        + tz_same_circuit_cost(p, q, r, cost_zz=cost_zz, cost_1q=cost_1q)
        + zzz_circuit_cost(cost_zz=cost_zz, cost_1q=cost_1q)
    )


def sz_opp_circuit(
    p: int,
    q: int,
    r: int,
    x: int,
    angle: float,
    n: int,
    offset: int = 0,
) -> str:
    """QASM for exp(i*angle * S_pqx Z_r^(1-x))."""
    s = ""
    s += z_circuit(r, 1 - x, angle / 2, n, offset)
    s += tz_opp_circuit(p, q, x, r, angle, n, offset)
    s += zzz_opp_circuit(p, q, r, x, angle / 2, n, offset)
    return s


def sz_opp_circuit_cost(
    p: int, q: int, *, cost_zz: float = 1.0, cost_1q: float = 0.0
) -> float:
    """Weighted cost for sz_opp_circuit: z_circuit + tz_opp_circuit + zzz_opp_circuit."""
    return (
        z_circuit_cost(cost_zz=cost_zz, cost_1q=cost_1q)
        + tz_opp_circuit_cost(p, q, cost_zz=cost_zz, cost_1q=cost_1q)
        + zzz_circuit_cost(cost_zz=cost_zz, cost_1q=cost_1q)
    )


def szz_same_nonoverlap_circuit(
    p: int,
    q: int,
    r: int,
    s_orb: int,
    x: int,
    angle: float,
    n: int,
    offset: int = 0,
) -> str:
    """QASM for exp(i*angle * S_pqx Z_rx Z_sx), both r,s outside (p,q).

    Parity trick: XOR Z_s onto Z_r so qr holds Z_r·Z_s, call sz_same on r,
    uncompute. Cost: 2 + sz_same_circuit_cost(p, q, r).
    """
    qr = qubit_index(r, x, n, offset)
    qs = qubit_index(s_orb, x, n, offset)
    s = ""
    s += f"cx q[{qs}],q[{qr}];\n"
    s += sz_same_circuit(p, q, r, x, angle, n, offset)
    s += f"cx q[{qs}],q[{qr}];\n"
    return s


def szz_same_interleaved_circuit(
    p: int,
    q: int,
    r: int,
    s_orb: int,
    x: int,
    angle: float,
    n: int,
    offset: int = 0,
) -> str:
    """QASM for exp(i*angle * S_pqx Z_rx Z_sx), exactly one of r,s inside (p,q).

    Explicit decomposition zz + tzz_same + zzzz: tzz_same exploits JW-string
    cancellation on the inside qubit, beating the parity trick by one gate.
    """
    s = ""
    s += zz_circuit(r, x, s_orb, x, angle / 2, n, offset)
    s += tzz_same_circuit(p, q, r, s_orb, x, angle, n, offset)
    s += zzzz_same_circuit(p, q, r, s_orb, x, angle / 2, n, offset)
    return s


def szz_same_nested_circuit(
    p: int,
    q: int,
    r: int,
    s_orb: int,
    x: int,
    angle: float,
    n: int,
    offset: int = 0,
) -> str:
    """QASM for exp(i*angle * S_pqx Z_rx Z_sx), both r,s inside (p,q).

    Explicit decomposition zz + tzz_same + zzzz. The parity trick fails here
    because the CNOT-conjugation would modify T_pq's JW string. tzz_same's
    eff-empty fallback (when both r,s coincide with intermediates and d=3)
    gives a particularly cheap path.
    """
    s = ""
    s += zz_circuit(r, x, s_orb, x, angle / 2, n, offset)
    s += tzz_same_circuit(p, q, r, s_orb, x, angle, n, offset)
    s += zzzz_same_circuit(p, q, r, s_orb, x, angle / 2, n, offset)
    return s


def szz_same_circuit(
    p: int,
    q: int,
    r: int,
    s_orb: int,
    x: int,
    angle: float,
    n: int,
    offset: int = 0,
) -> str:
    """QASM for exp(i*angle * S_pqx Z_rx Z_sx) with {p,q} ∩ {r,s} = ∅.

    Dispatches to nonoverlap / interleaved / nested based on whether each
    of r, s lies inside the qubit interval (p, q).
    """
    qp = qubit_index(p, x, n, offset)
    qq = qubit_index(q, x, n, offset)
    qr = qubit_index(r, x, n, offset)
    qs = qubit_index(s_orb, x, n, offset)
    lo, hi = min(qp, qq), max(qp, qq)
    inside = int(lo < qr < hi) + int(lo < qs < hi)
    if inside == 0:
        return szz_same_nonoverlap_circuit(p, q, r, s_orb, x, angle, n, offset)
    if inside == 1:
        return szz_same_interleaved_circuit(p, q, r, s_orb, x, angle, n, offset)
    return szz_same_nested_circuit(p, q, r, s_orb, x, angle, n, offset)


def szz_same_circuit_cost(
    p: int,
    q: int,
    r: int,
    s_orb: int,
    *,
    cost_zz: float = 1.0,
    cost_1q: float = 0.0,
) -> float:
    """Weighted cost for szz_same_circuit, dispatched by case.

    - Non-overlap (both outside): 2 CX (parity trick) + sz_same_circuit cost.
    - Interleaved (one inside): explicit zz_circuit + tzz_same_circuit + zzzz_same_circuit.
    - Nested (both inside): same composition as interleaved.
    """
    lo, hi = min(p, q), max(p, q)
    inside = int(lo < r < hi) + int(lo < s_orb < hi)
    if inside == 0:
        return 2 * cost_zz + sz_same_circuit_cost(
            p, q, r, cost_zz=cost_zz, cost_1q=cost_1q
        )
    return (
        zz_circuit_cost(cost_zz=cost_zz, cost_1q=cost_1q)
        + tzz_same_circuit_cost(p, q, r, s_orb, cost_zz=cost_zz, cost_1q=cost_1q)
        + zzzz_circuit_cost(cost_zz=cost_zz, cost_1q=cost_1q)
    )


def szz_opp_circuit(
    p: int,
    q: int,
    r: int,
    s_orb: int,
    x: int,
    angle: float,
    n: int,
    offset: int = 0,
) -> str:
    """QASM for exp(i*angle * S_pqx Z_r^(1-x) Z_s^(1-x)).

    Computes Z_r · Z_s parity onto qr via one CNOT, applies sz_opp_circuit
    (which sees a single Z on qr that now represents the parity), then
    uncomputes. Cost: 2 + sz_opp_circuit_cost(p, q).
    """
    qr = qubit_index(r, 1 - x, n, offset)
    qs = qubit_index(s_orb, 1 - x, n, offset)
    s = ""
    s += f"cx q[{qs}],q[{qr}];\n"
    s += sz_opp_circuit(p, q, r, x, angle, n, offset)
    s += f"cx q[{qs}],q[{qr}];\n"
    return s


def szz_opp_circuit_cost(
    p: int, q: int, *, cost_zz: float = 1.0, cost_1q: float = 0.0
) -> float:
    """Weighted cost for szz_opp_circuit: 2 CX (parity trick) + sz_opp_circuit cost."""
    return 2 * cost_zz + sz_opp_circuit_cost(p, q, cost_zz=cost_zz, cost_1q=cost_1q)


# TODO: ss_same_circuit and ss_opp_circuit (and their costs) are not yet
# implemented. The corresponding coefficient tensors (coeff_SS_same, coeff_SS_opp)
# already exist in ElectronicStructureSZ and contribute to its one_norm, but
# Trotter-step QASM synthesis for S_pq · S_rs products is still missing.


# ── TZ cost tensors ──────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=32)
def circuit_cost_tensors_tz(n: int, cost_zz: float = 1.0, cost_1q: float = 0.0) -> dict:
    """Per-term-group cost tensors for an n-orbital TZ Hamiltonian under JW.

    Each tensor is indexed by orbital positions in the JW string; entry
    [i, j, ...] is the gate cost when the term's orbitals land at positions
    (i, j, ...). Scalar entries (Z, ZZ) are position-independent.

    Results are cached by (n, cost_zz, cost_1q). Returned arrays are shared
    across calls — do not mutate them.
    """
    kw = dict(cost_zz=cost_zz, cost_1q=cost_1q)
    cost_T = np.zeros((n, n))
    cost_TZ_opp = np.zeros((n, n))
    cost_TZ_same = np.zeros((n, n, n))
    cost_TT_opp = np.zeros((n, n, n, n))
    cost_TT_same = np.zeros((n, n, n, n))
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            cost_T[p, q] = t_circuit_cost(p, q, **kw)
            cost_TZ_opp[p, q] = tz_opp_circuit_cost(p, q, **kw)
            for r in range(n):
                if r != p and r != q:
                    cost_TZ_same[p, q, r] = tz_same_circuit_cost(p, q, r, **kw)
                for s in range(n):
                    if s == r:
                        continue
                    cost_TT_opp[p, q, r, s] = tt_opp_circuit_cost(p, q, r, s, **kw)
                    if len({p, q, r, s}) == 4:
                        cost_TT_same[p, q, r, s] = tt_same_circuit_cost(
                            p, q, r, s, **kw
                        )
    return {
        "Z": z_circuit_cost(**kw),
        "ZZ": zz_circuit_cost(**kw),
        "T": cost_T,
        "TZ_opp": cost_TZ_opp,
        "TZ_same": cost_TZ_same,
        "TT_opp": cost_TT_opp,
        "TT_same": cost_TT_same,
    }


def term_costs_tz(n: int, cost_zz: float, cost_1q: float, terms: list) -> np.ndarray:
    """Gate cost for TZ terms; as given by ElectronicStructureTZ.enumerate_terms().
    Looked up from circuit_cost_tensors_tz output."""
    costs = []
    ct = circuit_cost_tensors_tz(n=n, cost_1q=cost_1q, cost_zz=cost_zz)
    for kind, indices, _spin, _, _ in terms:
        if kind == "Z":
            costs.append(float(ct["Z"]))
        elif kind in ("ZZ_same", "ZZ_opp"):
            costs.append(float(ct["ZZ"]))
        elif kind == "T":
            p, q = indices
            costs.append(float(ct["T"][p, q]))
        elif kind == "TZ_opp":
            p, q, _r = indices
            costs.append(float(ct["TZ_opp"][p, q]))
        elif kind == "TZ_same":
            p, q, r = indices
            costs.append(float(ct["TZ_same"][p, q, r]))
        elif kind == "TT_opp":
            p, q, r, s = indices
            costs.append(float(ct["TT_opp"][p, q, r, s]))
        elif kind == "TT_same":
            p, q, r, s = indices
            costs.append(float(ct["TT_same"][p, q, r, s]))
        else:
            raise ValueError(f"Unknown TZ term kind '{kind}'")
    return np.array(costs)
