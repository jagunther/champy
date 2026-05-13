"""Tests for ElectronicStructureSZ and the TZ -> SZ conversion.

The conversion correctness is verified by re-expanding the SZ representation
back to a PauliHamiltonian (using S_pq = (X_p P X_q + Y_p P Y_q + I + Z_p Z_q)/2)
and comparing the resulting matrix to the one produced by the TZ -> Pauli
conversion of the same input.
"""

from itertools import product

import numpy as np
import pytest

from champy.ElectronicStructure import ElectronicStructure
from champy.ElectronicStructureSZ import ElectronicStructureSZ
from champy.ElectronicStructureTZ import ElectronicStructureTZ
from champy import circuits
from champy.PauliHamiltonian import PauliHamiltonian, multiply_labels
from champy.tests.test_ElectronicStructureTZ import (
    _kron,
    _simulate_qasm,
    _I,
    _Z,
    _t_operator,
)


def _sz_to_pauli(sz) -> PauliHamiltonian:
    """Expand each SZ term into Paulis (S_pq -> 4 Paulis) and assemble."""
    n = sz.num_orb
    nq = 2 * n
    labels: list[str] = []
    weights: list[float] = []

    def lbl(ops: dict) -> str:
        chars = ["I"] * nq
        for q, p in ops.items():
            chars[q] = p
        return "".join(chars)

    id_label = "I" * nq

    def s_factors(p: int, q: int, off: int):
        """4-Pauli decomposition of S_pq on the spin sector with offset `off`."""
        pp, qq = p + off, q + off
        chars_xx = ["I"] * nq
        chars_yy = ["I"] * nq
        chars_xx[pp] = "X"
        chars_yy[pp] = "Y"
        for k in range(pp + 1, qq):
            chars_xx[k] = "Z"
            chars_yy[k] = "Z"
        chars_xx[qq] = "X"
        chars_yy[qq] = "Y"
        return [
            ("".join(chars_xx), 0.5),
            ("".join(chars_yy), 0.5),
            (id_label, 0.5),
            (lbl({pp: "Z", qq: "Z"}), 0.5),
        ]

    def add_product(factor_lists, weight):
        """Distribute the product of operator-decompositions and accumulate."""
        for combo in product(*factor_lists):
            label = combo[0][0]
            w = combo[0][1]
            for sublabel, sw in combo[1:]:
                label, phase = multiply_labels(label, sublabel)
                w *= sw * phase
            labels.append(label)
            weights.append(weight * w)

    # constant
    labels.append(id_label)
    weights.append(sz._constant)

    # Z_p (per-orbital, both spins)
    for p in range(n):
        c = sz.coeff_Z[p]
        if c == 0:
            continue
        for off in (0, n):
            labels.append(lbl({p + off: "Z"}))
            weights.append(c)

    # ZZ_same (p<q, both spins)
    for p in range(n):
        for q in range(p + 1, n):
            c = sz.coeff_ZZ_same[p, q]
            if c == 0:
                continue
            for off in (0, n):
                labels.append(lbl({p + off: "Z", q + off: "Z"}))
                weights.append(c)

    # ZZ_opp
    for p in range(n):
        for q in range(n):
            c = sz.coeff_ZZ_opp[p, q]
            if c == 0:
                continue
            labels.append(lbl({p: "Z", q + n: "Z"}))
            weights.append(c)

    # ZZZ_opp ((p,q) one spin, r other; both spin orderings)
    for p in range(n):
        for q in range(p + 1, n):
            for r in range(n):
                c = sz.coeff_ZZZ_opp[p, q, r]
                if c == 0:
                    continue
                labels.append(lbl({p: "Z", q: "Z", r + n: "Z"}))
                weights.append(c)
                labels.append(lbl({p + n: "Z", q + n: "Z", r: "Z"}))
                weights.append(c)

    # ZZZ_same (p<q, r∉{p,q}, both spins)
    for p in range(n):
        for q in range(p + 1, n):
            for r in range(n):
                if r in (p, q):
                    continue
                c = sz.coeff_ZZZ_same[p, q, r]
                if c == 0:
                    continue
                for off in (0, n):
                    labels.append(
                        lbl({p + off: "Z", q + off: "Z", r + off: "Z"})
                    )
                    weights.append(c)

    # ZZZZ_opp (p<q, r<s; convention: alpha pair = (p,q), beta pair = (r,s))
    # Spin-flipped operator is stored at swapped index [r,s,p,q].
    for p in range(n):
        for q in range(p + 1, n):
            for r in range(n):
                for s in range(r + 1, n):
                    c = sz.coeff_ZZZZ_opp[p, q, r, s]
                    if c == 0:
                        continue
                    labels.append(
                        lbl({p: "Z", q: "Z", r + n: "Z", s + n: "Z"})
                    )
                    weights.append(c)

    # ZZZZ_same (canonical: p<q, r<s, p<r, disjoint; both spins)
    for p in range(n):
        for q in range(p + 1, n):
            for r in range(p + 1, n):
                for s in range(r + 1, n):
                    if {p, q} & {r, s}:
                        continue
                    c = sz.coeff_ZZZZ_same[p, q, r, s]
                    if c == 0:
                        continue
                    for off in (0, n):
                        labels.append(
                            lbl(
                                {
                                    p + off: "Z",
                                    q + off: "Z",
                                    r + off: "Z",
                                    s + off: "Z",
                                }
                            )
                        )
                        weights.append(c)

    # S_pq (p<q, both spins)
    for p in range(n):
        for q in range(p + 1, n):
            c = sz.coeff_S[p, q]
            if c == 0:
                continue
            for off in (0, n):
                add_product([s_factors(p, q, off)], c)

    # SZ_opp ((p,q) one spin, r other spin)
    for p in range(n):
        for q in range(p + 1, n):
            for r in range(n):
                c = sz.coeff_SZ_opp[p, q, r]
                if c == 0:
                    continue
                add_product(
                    [s_factors(p, q, 0), [(lbl({r + n: "Z"}), 1.0)]], c
                )
                add_product(
                    [s_factors(p, q, n), [(lbl({r: "Z"}), 1.0)]], c
                )

    # SZ_same (p<q, r∉{p,q}, both spins)
    for p in range(n):
        for q in range(p + 1, n):
            for r in range(n):
                if r in (p, q):
                    continue
                c = sz.coeff_SZ_same[p, q, r]
                if c == 0:
                    continue
                for off in (0, n):
                    add_product(
                        [s_factors(p, q, off), [(lbl({r + off: "Z"}), 1.0)]],
                        c,
                    )

    # SS_opp (convention: alpha-pair = (p,q), beta-pair = (r,s)).
    # Spin-flipped operator is stored at swapped index [r,s,p,q].
    for p in range(n):
        for q in range(p + 1, n):
            for r in range(n):
                for s in range(r + 1, n):
                    c = sz.coeff_SS_opp[p, q, r, s]
                    if c == 0:
                        continue
                    add_product([s_factors(p, q, 0), s_factors(r, s, n)], c)

    # SS_same (canonical: p<q, r<s, p<r, disjoint; both spins)
    for p in range(n):
        for q in range(p + 1, n):
            for r in range(p + 1, n):
                for s in range(r + 1, n):
                    if {p, q} & {r, s}:
                        continue
                    c = sz.coeff_SS_same[p, q, r, s]
                    if c == 0:
                        continue
                    for off in (0, n):
                        add_product(
                            [s_factors(p, q, off), s_factors(r, s, off)], c
                        )

    # SZZ_opp (S on (p,q) one spin, ZZ on (r,s) other)
    for p in range(n):
        for q in range(p + 1, n):
            for r in range(n):
                for s in range(r + 1, n):
                    c = sz.coeff_SZZ_opp[p, q, r, s]
                    if c == 0:
                        continue
                    add_product(
                        [
                            s_factors(p, q, 0),
                            [(lbl({r + n: "Z", s + n: "Z"}), 1.0)],
                        ],
                        c,
                    )
                    add_product(
                        [
                            s_factors(p, q, n),
                            [(lbl({r: "Z", s: "Z"}), 1.0)],
                        ],
                        c,
                    )

    # SZZ_same (S on (p,q), ZZ on (r,s); p<q, r<s, disjoint; both spins)
    for p in range(n):
        for q in range(p + 1, n):
            for r in range(n):
                for s in range(r + 1, n):
                    if {p, q} & {r, s}:
                        continue
                    c = sz.coeff_SZZ_same[p, q, r, s]
                    if c == 0:
                        continue
                    for off in (0, n):
                        add_product(
                            [
                                s_factors(p, q, off),
                                [(lbl({r + off: "Z", s + off: "Z"}), 1.0)],
                            ],
                            c,
                        )

    return PauliHamiltonian.from_labels_and_weights(labels, weights)


def _matrix(pauli: PauliHamiltonian) -> np.ndarray:
    return pauli.to_sparse_matrix().toarray() + pauli.constant * np.eye(
        pauli.dimension
    )


@pytest.mark.parametrize("hamil_random", [(3, 2), (4, 4)], indirect=True)
def test_tz_to_sz_matches_pauli(hamil_random):
    """TZ.to_ElectronicStructureSZ() preserves the Hamiltonian matrix."""
    elstruc = hamil_random
    tz = ElectronicStructureTZ(elstruc.h0, elstruc.h1e, elstruc.h2e, elstruc.num_elec)
    sz = tz.to_ElectronicStructureSZ()

    M_tz = _matrix(tz.to_pauli_hamiltonian())
    M_sz = _matrix(_sz_to_pauli(sz))

    assert np.allclose(M_tz, M_sz, atol=1e-5, rtol=1e-5), (
        f"TZ and SZ Pauli matrices differ; max |Δ| = "
        f"{np.max(np.abs(M_tz - M_sz)):.3e}"
    )


def test_sz_constructor_runs():
    """The skeleton SZ class instantiates with zero coefficients."""
    from champy.ElectronicStructureSZ import ElectronicStructureSZ

    n = 3
    h1e = np.zeros((n, n))
    h2e = np.zeros((n, n, n, n))
    sz = ElectronicStructureSZ(0.0, h1e, h2e, num_elec=2)
    assert sz.num_orb == n
    assert sz.constant == 0.0
    assert np.all(sz.coeff_S == 0)
    assert np.all(sz.coeff_SS_opp == 0)


def test_one_norm_per_term_multiplicity():
    """one_norm respects the spin multiplicity for each term type."""
    from champy.ElectronicStructureSZ import ElectronicStructureSZ

    n = 4
    sz = ElectronicStructureSZ(
        0.0, np.zeros((n, n)), np.zeros((n, n, n, n)), num_elec=2
    )

    # ×2 (same-spin) entries:
    sz.coeff_Z[0] = 1.5
    sz.coeff_S[0, 1] = 2.0
    sz.coeff_ZZ_same[0, 1] = 0.5
    sz.coeff_SZ_same[0, 1, 2] = 1.0
    sz.coeff_ZZZ_same[0, 1, 2] = 0.25
    sz.coeff_SS_same[0, 1, 2, 3] = 0.75
    sz.coeff_SZZ_same[0, 1, 2, 3] = 0.5
    sz.coeff_ZZZZ_same[0, 1, 2, 3] = 0.125
    # ×2 (opp-spin, mixed-type halves) entries:
    sz.coeff_SZ_opp[0, 1, 2] = 0.4
    sz.coeff_ZZZ_opp[0, 1, 2] = 0.3
    sz.coeff_SZZ_opp[0, 1, 2, 3] = 0.2
    # ×1 (opp-spin, same-type halves) entries:
    sz.coeff_ZZ_opp[0, 1] = 1.0
    sz.coeff_SS_opp[0, 1, 2, 3] = 1.5
    sz.coeff_ZZZZ_opp[0, 1, 2, 3] = 0.625

    expected = (
        2 * (1.5 + 2.0 + 0.5 + 1.0 + 0.25 + 0.75 + 0.5 + 0.125)  # same-spin
        + 2 * (0.4 + 0.3 + 0.2)  # opp-spin mixed
        + 1 * (1.0 + 1.5 + 0.625)  # opp-spin same-type
    )
    assert sz.one_norm() == pytest.approx(expected)


@pytest.mark.parametrize("hamil_random", [(4, 4)], indirect=True)
def test_one_norm_finite_after_conversion(hamil_random):
    """Converting a random TZ to SZ produces a finite, positive one_norm."""
    elstruc = hamil_random
    tz = ElectronicStructureTZ(elstruc.h0, elstruc.h1e, elstruc.h2e, elstruc.num_elec)
    norm = tz.to_ElectronicStructureSZ().one_norm()
    assert np.isfinite(norm) and norm > 0


# ── Z-only QASM circuits ────────────────────────────────────────────────────


def _z_product(qubit_indices, n_qubits):
    return _kron(*[_Z if i in qubit_indices else _I for i in range(n_qubits)])


def _check_z_circuit(qasm: str, qubit_indices, n_qubits: int, angle: float):
    from scipy.linalg import expm

    circuit = _simulate_qasm(qasm, n_qubits)
    target = expm(1j * angle * _z_product(qubit_indices, n_qubits))
    assert np.allclose(circuit, target)


@pytest.mark.parametrize("p,x", [(1, 0), (3, 0), (2, 1)])
def test_z_circuit(p, x):
    n_orb = 4
    n_qubits = 2 * n_orb
    angle = 0.31
    qasm = circuits.z_circuit(p, x, angle, n_orb)
    qp = circuits.qubit_index(p, x, n_orb, 0)
    _check_z_circuit(qasm, [qp], n_qubits, angle)


@pytest.mark.parametrize("p,q,x", [(1, 2, 0), (1, 4, 0), (2, 3, 1)])
def test_zz_same_circuit(p, q, x):
    n_orb = 4
    n_qubits = 2 * n_orb
    angle = 0.31
    qasm = circuits.zz_circuit(p, x, q, x, angle, n_orb)
    qp = circuits.qubit_index(p, x, n_orb, 0)
    qq = circuits.qubit_index(q, x, n_orb, 0)
    _check_z_circuit(qasm, [qp, qq], n_qubits, angle)


@pytest.mark.parametrize("p,q", [(1, 1), (2, 3), (4, 1)])
def test_zz_opp_circuit(p, q):
    n_orb = 4
    n_qubits = 2 * n_orb
    angle = 0.31
    qasm = circuits.zz_circuit(p, 0, q, 1, angle, n_orb)
    qp = circuits.qubit_index(p, 0, n_orb, 0)
    qq = circuits.qubit_index(q, 1, n_orb, 0)
    _check_z_circuit(qasm, [qp, qq], n_qubits, angle)


@pytest.mark.parametrize("p,q,r,x", [(1, 2, 3, 0), (1, 3, 4, 0), (2, 3, 4, 1)])
def test_zzz_same_circuit(p, q, r, x):
    n_orb = 4
    n_qubits = 2 * n_orb
    angle = 0.31
    qasm = circuits.zzz_same_circuit(p, q, r, x, angle, n_orb)
    qubits = [circuits.qubit_index(o, x, n_orb, 0) for o in (p, q, r)]
    _check_z_circuit(qasm, qubits, n_qubits, angle)


@pytest.mark.parametrize(
    "p,q,r,x", [(1, 2, 3, 0), (1, 2, 1, 0), (2, 4, 3, 1)]
)
def test_zzz_opp_circuit(p, q, r, x):
    n_orb = 4
    n_qubits = 2 * n_orb
    angle = 0.31
    qasm = circuits.zzz_opp_circuit(p, q, r, x, angle, n_orb)
    qubits = [
        circuits.qubit_index(p, x, n_orb, 0),
        circuits.qubit_index(q, x, n_orb, 0),
        circuits.qubit_index(r, 1 - x, n_orb, 0),
    ]
    _check_z_circuit(qasm, qubits, n_qubits, angle)


@pytest.mark.parametrize("p,q,r,s,x", [(1, 2, 3, 4, 0), (1, 2, 3, 4, 1)])
def test_zzzz_same_circuit(p, q, r, s, x):
    n_orb = 4
    n_qubits = 2 * n_orb
    angle = 0.31
    qasm = circuits.zzzz_same_circuit(p, q, r, s, x, angle, n_orb)
    qubits = [
        circuits.qubit_index(o, x, n_orb, 0) for o in (p, q, r, s)
    ]
    _check_z_circuit(qasm, qubits, n_qubits, angle)


@pytest.mark.parametrize("p,q,r,s", [(1, 2, 1, 2), (1, 3, 2, 4), (2, 4, 1, 3)])
def test_zzzz_opp_circuit(p, q, r, s):
    n_orb = 4
    n_qubits = 2 * n_orb
    angle = 0.31
    qasm = circuits.zzzz_opp_circuit(p, q, r, s, angle, n_orb)
    qubits = [
        circuits.qubit_index(p, 0, n_orb, 0),
        circuits.qubit_index(q, 0, n_orb, 0),
        circuits.qubit_index(r, 1, n_orb, 0),
        circuits.qubit_index(s, 1, n_orb, 0),
    ]
    _check_z_circuit(qasm, qubits, n_qubits, angle)


# ── S-containing QASM circuits ───────────────────────────────────────────────


def _s_operator(qp: int, qq: int, n_qubits: int):
    """S_pq = T_pq + (I + Z_p Z_q)/2 as a dense matrix on n_qubits qubits."""
    T = _t_operator(qp, qq, n_qubits)
    Z_pZ_q = _kron(*[_Z if i in (qp, qq) else _I for i in range(n_qubits)])
    Id = np.eye(2**n_qubits)
    return T + (Id + Z_pZ_q) / 2


@pytest.mark.parametrize("p,q,x", [(1, 2, 0), (1, 3, 0), (1, 4, 0), (2, 4, 1)])
def test_s_circuit(p, q, x):
    from scipy.linalg import expm

    n_orb = 4
    n_qubits = 2 * n_orb
    angle = 0.27
    qasm = circuits.s_circuit(p, q, x, angle, n_orb)
    circuit = _simulate_qasm(qasm, n_qubits)
    qp = circuits.qubit_index(p, x, n_orb, 0)
    qq = circuits.qubit_index(q, x, n_orb, 0)
    target = expm(1j * angle * _s_operator(qp, qq, n_qubits))
    # Drop the I/2 part's global phase: e^{iθ/2}
    target = target * np.exp(-1j * angle / 2)
    assert np.allclose(circuit, target), (
        f"s_circuit failed for p={p},q={q},x={x}"
    )


@pytest.mark.parametrize(
    "p,q,r,x",
    [(1, 2, 3, 0), (1, 3, 2, 0), (1, 4, 2, 0), (2, 4, 1, 1)],
)
def test_sz_same_circuit(p, q, r, x):
    from scipy.linalg import expm

    n_orb = 4
    n_qubits = 2 * n_orb
    angle = 0.27
    qasm = circuits.sz_same_circuit(p, q, r, x, angle, n_orb)
    circuit = _simulate_qasm(qasm, n_qubits)
    qp = circuits.qubit_index(p, x, n_orb, 0)
    qq = circuits.qubit_index(q, x, n_orb, 0)
    qr = circuits.qubit_index(r, x, n_orb, 0)
    S = _s_operator(qp, qq, n_qubits)
    Z_r = _kron(*[_Z if i == qr else _I for i in range(n_qubits)])
    target = expm(1j * angle * S @ Z_r)
    assert np.allclose(circuit, target), (
        f"sz_same_circuit failed for p={p},q={q},r={r},x={x}"
    )


@pytest.mark.parametrize(
    "p,q,r,x",
    [(1, 2, 1, 0), (1, 3, 2, 0), (1, 4, 2, 0), (2, 4, 1, 1)],
)
def test_sz_opp_circuit(p, q, r, x):
    from scipy.linalg import expm

    n_orb = 4
    n_qubits = 2 * n_orb
    angle = 0.27
    qasm = circuits.sz_opp_circuit(p, q, r, x, angle, n_orb)
    circuit = _simulate_qasm(qasm, n_qubits)
    qp = circuits.qubit_index(p, x, n_orb, 0)
    qq = circuits.qubit_index(q, x, n_orb, 0)
    qr = circuits.qubit_index(r, 1 - x, n_orb, 0)
    S = _s_operator(qp, qq, n_qubits)
    Z_r = _kron(*[_Z if i == qr else _I for i in range(n_qubits)])
    target = expm(1j * angle * S @ Z_r)
    assert np.allclose(circuit, target), (
        f"sz_opp_circuit failed for p={p},q={q},r={r},x={x}"
    )


# ── TZZ and SZZ circuits ─────────────────────────────────────────────────────


def _zstring_op(qubits, n_qubits):
    return _kron(*[_Z if i in qubits else _I for i in range(n_qubits)])


@pytest.mark.parametrize(
    "p,q,r,s,x",
    [
        (1, 2, 3, 4, 0),  # d=1, both r,s outside
        (1, 4, 2, 3, 0),  # d=3, both r,s inside (full cancellation)
        (1, 4, 2, 5, 0),  # d=3, one inside one outside
        (1, 3, 4, 5, 0),  # d=2, both outside
        (2, 5, 1, 4, 1),  # spin=1
    ],
)
def test_tzz_same_circuit(p, q, r, s, x):
    from scipy.linalg import expm

    n_orb = max(p, q, r, s) + 1
    n_qubits = 2 * n_orb
    angle = 0.21
    qasm = circuits.tzz_same_circuit(
        p, q, r, s, x, angle, n_orb
    )
    circuit = _simulate_qasm(qasm, n_qubits)
    qp = circuits.qubit_index(p, x, n_orb, 0)
    qq = circuits.qubit_index(q, x, n_orb, 0)
    qr = circuits.qubit_index(r, x, n_orb, 0)
    qs = circuits.qubit_index(s, x, n_orb, 0)
    T = _t_operator(qp, qq, n_qubits)
    Z_rs = _zstring_op({qr, qs}, n_qubits)
    target = expm(1j * angle * T @ Z_rs)
    assert np.allclose(circuit, target), (
        f"tzz_same_circuit failed for p={p},q={q},r={r},s={s},x={x}"
    )


@pytest.mark.parametrize(
    "p,q,r,s,x",
    [
        (1, 2, 1, 2, 0),
        (1, 3, 2, 4, 0),
        (1, 4, 1, 3, 0),
        (2, 4, 1, 3, 1),
    ],
)
def test_tzz_opp_circuit(p, q, r, s, x):
    from scipy.linalg import expm

    n_orb = max(p, q, r, s) + 1
    n_qubits = 2 * n_orb
    angle = 0.21
    qasm = circuits.tzz_opp_circuit(
        p, q, r, s, x, angle, n_orb
    )
    circuit = _simulate_qasm(qasm, n_qubits)
    qp = circuits.qubit_index(p, x, n_orb, 0)
    qq = circuits.qubit_index(q, x, n_orb, 0)
    qr = circuits.qubit_index(r, 1 - x, n_orb, 0)
    qs = circuits.qubit_index(s, 1 - x, n_orb, 0)
    T = _t_operator(qp, qq, n_qubits)
    Z_rs = _zstring_op({qr, qs}, n_qubits)
    target = expm(1j * angle * T @ Z_rs)
    assert np.allclose(circuit, target), (
        f"tzz_opp_circuit failed for p={p},q={q},r={r},s={s},x={x}"
    )


@pytest.mark.parametrize(
    "p,q,r,s,x",
    [
        (1, 2, 3, 4, 0),
        (1, 4, 2, 3, 0),  # both inside
        (1, 4, 2, 5, 0),
        (1, 3, 4, 5, 0),
        (2, 5, 1, 4, 1),
    ],
)
def test_szz_same_circuit(p, q, r, s, x):
    from scipy.linalg import expm

    n_orb = max(p, q, r, s) + 1
    n_qubits = 2 * n_orb
    angle = 0.21
    qasm = circuits.szz_same_circuit(
        p, q, r, s, x, angle, n_orb
    )
    circuit = _simulate_qasm(qasm, n_qubits)
    qp = circuits.qubit_index(p, x, n_orb, 0)
    qq = circuits.qubit_index(q, x, n_orb, 0)
    qr = circuits.qubit_index(r, x, n_orb, 0)
    qs = circuits.qubit_index(s, x, n_orb, 0)
    S = _s_operator(qp, qq, n_qubits)
    Z_rs = _zstring_op({qr, qs}, n_qubits)
    target = expm(1j * angle * S @ Z_rs)
    assert np.allclose(circuit, target), (
        f"szz_same_circuit failed for p={p},q={q},r={r},s={s},x={x}"
    )


@pytest.mark.parametrize(
    "p,q,r,s,x",
    [
        (1, 2, 1, 2, 0),
        (1, 3, 2, 4, 0),
        (1, 4, 1, 3, 0),
        (2, 4, 1, 3, 1),
    ],
)
def test_szz_opp_circuit(p, q, r, s, x):
    from scipy.linalg import expm

    n_orb = max(p, q, r, s) + 1
    n_qubits = 2 * n_orb
    angle = 0.21
    qasm = circuits.szz_opp_circuit(
        p, q, r, s, x, angle, n_orb
    )
    circuit = _simulate_qasm(qasm, n_qubits)
    qp = circuits.qubit_index(p, x, n_orb, 0)
    qq = circuits.qubit_index(q, x, n_orb, 0)
    qr = circuits.qubit_index(r, 1 - x, n_orb, 0)
    qs = circuits.qubit_index(s, 1 - x, n_orb, 0)
    S = _s_operator(qp, qq, n_qubits)
    Z_rs = _zstring_op({qr, qs}, n_qubits)
    target = expm(1j * angle * S @ Z_rs)
    assert np.allclose(circuit, target), (
        f"szz_opp_circuit failed for p={p},q={q},r={r},s={s},x={x}"
    )
