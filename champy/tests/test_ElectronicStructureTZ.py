import pytest
import numpy as np
from champy.ElectronicStructure import ElectronicStructure
from champy.ElectronicStructureTZ import ElectronicStructureTZ
from champy.PauliHamiltonian import PauliHamiltonian


def _f2q_valid(elstruc: ElectronicStructure, pauli_hamil: PauliHamiltonian):
    spec_elstruc = np.linalg.eigvals(elstruc.to_sparse_matrix()) + elstruc.constant
    spec_pauli = np.real(
        np.linalg.eigvals(pauli_hamil.to_sparse_matrix().toarray())
        + pauli_hamil.constant
    )

    for e in spec_elstruc:
        if not np.any(np.isclose(spec_pauli, e, atol=1e-6)):
            return False, e
    return True, None


@pytest.mark.parametrize("hamil_random", [(4, 4)], indirect=True)
def test_to_pauli_hamiltonian(hamil_random):
    elstruc = hamil_random
    tz = ElectronicStructureTZ(elstruc.h0, elstruc.h1e, elstruc.h2e)
    pauli_hamil = tz.to_pauli_hamiltonian()
    valid, error = _f2q_valid(elstruc=elstruc, pauli_hamil=pauli_hamil)
    assert valid, f"Eigenvalue {error} not found in Pauli spectrum"


# --- QASM simulation helpers ---

_I = np.eye(2)
_X = np.array([[0, 1], [1, 0]])
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1, -1])


def _kron(*args):
    r = args[0]
    for a in args[1:]:
        r = np.kron(r, a)
    return r


def _simulate_qasm(qasm_circuit_str, n_qubits):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from champy.ElectronicStructureTZ import QASM_GATE_DEFS

    qasm_prog = (
        f'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        f"{QASM_GATE_DEFS}qreg q[{n_qubits}];\n{qasm_circuit_str}"
    )
    qc = QuantumCircuit.from_qasm_str(qasm_prog)
    return Operator(qc.reverse_bits()).data


def _t_operator(p_qubit, q_qubit, n_qubits):
    ops_xx = [_I] * n_qubits
    ops_yy = [_I] * n_qubits
    ops_xx[p_qubit] = _X
    ops_xx[q_qubit] = _X
    ops_yy[p_qubit] = _Y
    ops_yy[q_qubit] = _Y
    for k in range(p_qubit + 1, q_qubit):
        ops_xx[k] = _Z
        ops_yy[k] = _Z
    return (_kron(*ops_xx) + _kron(*ops_yy)) / 2


@pytest.mark.parametrize("d", [1, 2, 3])
def test_t_circuit(d):
    from scipy.linalg import expm

    angle = 0.3
    p, q = 1, 1 + d
    n_orb = d + 1
    n_qubits = 2 * n_orb
    qasm = ElectronicStructureTZ.t_circuit(p, q, 0, angle, n_orb)
    circuit = _simulate_qasm(qasm, n_qubits)
    qp = ElectronicStructureTZ._qubit(p, 0, n_orb, 0)
    qq = ElectronicStructureTZ._qubit(q, 0, n_orb, 0)
    target = expm(1j * angle * _t_operator(qp, qq, n_qubits))
    assert np.allclose(circuit, target), f"t_circuit failed for d={d}"


@pytest.mark.parametrize(
    "p,q,r,x", [(1, 2, 1, 0), (1, 3, 2, 0), (1, 4, 1, 0), (2, 4, 3, 1), (1, 3, 4, 0)]
)
def test_tz_opp_circuit(p, q, r, x):
    from scipy.linalg import expm

    angle = 0.3
    n_orb = max(p, q, r) + 1
    n_qubits = 2 * n_orb
    qasm = ElectronicStructureTZ.tz_opp_circuit(p, q, x, r, angle, n_orb)
    circuit = _simulate_qasm(qasm, n_qubits)
    qp = ElectronicStructureTZ._qubit(p, x, n_orb, 0)
    qq = ElectronicStructureTZ._qubit(q, x, n_orb, 0)
    qr = ElectronicStructureTZ._qubit(r, 1 - x, n_orb, 0)
    T = _t_operator(qp, qq, n_qubits)
    Z_r = _kron(*[_Z if i == qr else _I for i in range(n_qubits)])
    target = expm(1j * angle * T @ Z_r)
    assert np.allclose(circuit, target), f"tz_opp_circuit failed for p={p},q={q},r={r}"


@pytest.mark.parametrize(
    "p,q,r,x",
    [
        (1, 2, 3, 0),  # d=1, r after q
        (2, 3, 1, 0),  # d=1, r before p
        (1, 3, 2, 0),  # r inside, d=2 (Givens)
        (1, 4, 2, 0),  # r inside, d=3
        (1, 4, 3, 0),  # r inside, d=3
        (2, 4, 1, 0),  # r before p
        (1, 3, 4, 0),  # r after q
    ],
)
def test_tz_same_circuit(p, q, r, x):
    from scipy.linalg import expm

    angle = 0.3
    n_orb = max(p, q, r) + 1
    n_qubits = 2 * n_orb
    qasm = ElectronicStructureTZ.tz_same_circuit(p, q, r, x, angle, n_orb)
    circuit = _simulate_qasm(qasm, n_qubits)
    qp = ElectronicStructureTZ._qubit(p, x, n_orb, 0)
    qq = ElectronicStructureTZ._qubit(q, x, n_orb, 0)
    qr = ElectronicStructureTZ._qubit(r, x, n_orb, 0)
    T = _t_operator(qp, qq, n_qubits)
    Z_r = _kron(*[_Z if i == qr else _I for i in range(n_qubits)])
    target = expm(1j * angle * T @ Z_r)
    assert np.allclose(circuit, target), f"tz_same_circuit failed for p={p},q={q},r={r}"


@pytest.mark.parametrize(
    "p,q,r,s", [(1, 3, 1, 3), (1, 4, 1, 3), (1, 3, 2, 4)]
)
def test_tt_opp_circuit(p, q, r, s):
    from scipy.linalg import expm

    angle = 0.3
    n_orb = max(p, q, r, s) + 1
    n_qubits = 2 * n_orb
    qasm = ElectronicStructureTZ.tt_opp_circuit(p, q, r, s, angle, n_orb)
    circuit = _simulate_qasm(qasm, n_qubits)
    qp = ElectronicStructureTZ._qubit(p, 0, n_orb, 0)
    qq = ElectronicStructureTZ._qubit(q, 0, n_orb, 0)
    qr = ElectronicStructureTZ._qubit(r, 1, n_orb, 0)
    qs = ElectronicStructureTZ._qubit(s, 1, n_orb, 0)
    Tu = _t_operator(qp, qq, n_qubits)
    Td = _t_operator(qr, qs, n_qubits)
    target = expm(1j * angle * Tu @ Td)
    assert np.allclose(circuit, target), f"tt_opp_circuit failed for p={p},q={q},r={r},s={s}"


@pytest.mark.parametrize(
    "p,q,r,s",
    [
        (1, 2, 3, 4),  # non-overlapping, d1=1, d2=1
        (1, 3, 4, 6),  # non-overlapping, d1=2, d2=2
        (1, 3, 4, 5),  # non-overlapping, d1=2, d2=1
    ],
)
def test_tt_same_nonoverlap_circuit(p, q, r, s):
    from scipy.linalg import expm

    angle = 0.3
    x = 0
    n_orb = max(p, q, r, s) + 1
    n_qubits = n_orb
    qasm = ElectronicStructureTZ.tt_same_nonoverlap_circuit(
        p, q, r, s, x, angle, n_orb
    )
    circuit = _simulate_qasm(qasm, n_qubits)
    qp = p - 1
    qq = q - 1
    qr = r - 1
    qs = s - 1
    T1 = _t_operator(qp, qq, n_qubits)
    T2 = _t_operator(qr, qs, n_qubits)
    target = expm(1j * angle * T1 @ T2)
    assert np.allclose(circuit, target), f"tt_same_nonoverlap_circuit failed for p={p},q={q},r={r},s={s}"


@pytest.mark.parametrize(
    "p,q,r,s",
    [
        (1, 4, 2, 3),  # p<r<s<q, d_outer=3, d_inner=1
        (1, 5, 2, 4),  # p<r<s<q, d_outer=4, d_inner=2
        (1, 5, 3, 4),  # p<r<s<q, d_outer=4, d_inner=1
        (1, 6, 2, 5),  # p<r<s<q, d_outer=5, d_inner=3
        (2, 4, 1, 5),  # r<p<q<s
        (3, 5, 1, 6),  # r<p<q<s
    ],
)
def test_tt_same_nested_circuit(p, q, r, s):
    from scipy.linalg import expm

    angle = 0.3
    x = 0
    n_orb = max(p, q, r, s) + 1
    n_qubits = n_orb
    qasm = ElectronicStructureTZ.tt_same_nested_circuit(
        p, q, r, s, x, angle, n_orb
    )
    circuit = _simulate_qasm(qasm, n_qubits)
    qp = p - 1
    qq = q - 1
    qr = r - 1
    qs = s - 1
    T1 = _t_operator(qp, qq, n_qubits)
    T2 = _t_operator(qr, qs, n_qubits)
    target = expm(1j * angle * T1 @ T2)
    assert np.allclose(circuit, target), f"tt_same_nested_circuit failed for p={p},q={q},r={r},s={s}"


@pytest.mark.parametrize(
    "p,q,r,s",
    [
        (1, 3, 2, 4),  # p<r<q<s
        (1, 4, 2, 5),  # p<r<q<s
        (1, 4, 3, 6),  # p<r<q<s
        (1, 5, 3, 7),  # p<r<q<s
        (2, 4, 1, 3),  # r<p<s<q
        (2, 5, 1, 4),  # r<p<s<q
        (3, 6, 1, 4),  # r<p<s<q
    ],
)
def test_tt_same_interleaved_circuit(p, q, r, s):
    from scipy.linalg import expm

    angle = 0.3
    x = 0
    n_orb = max(p, q, r, s) + 1
    n_qubits = n_orb
    qasm = ElectronicStructureTZ.tt_same_interleaved_circuit(
        p, q, r, s, x, angle, n_orb
    )
    circuit = _simulate_qasm(qasm, n_qubits)
    qp = p - 1
    qq = q - 1
    qr = r - 1
    qs = s - 1
    T1 = _t_operator(qp, qq, n_qubits)
    T2 = _t_operator(qr, qs, n_qubits)
    target = expm(1j * angle * T1 @ T2)
    assert np.allclose(circuit, target), f"tt_same_interleaved_circuit failed for p={p},q={q},r={r},s={s}"


# ── ElectronicStructure.to_ElectronicStructureTZ() ─────────────────────────


@pytest.mark.parametrize("hamil_random", [(4, 4)], indirect=True)
def test_to_ElectronicStructureTZ(hamil_random):
    """to_ElectronicStructureTZ() yields a TZ Hamiltonian whose Pauli image
    contains the original FCI spectrum."""
    elstruc = hamil_random
    tz = elstruc.to_ElectronicStructureTZ()
    assert isinstance(tz, ElectronicStructureTZ)
    pauli_hamil = tz.to_pauli_hamiltonian()
    valid, error = _f2q_valid(elstruc=elstruc, pauli_hamil=pauli_hamil)
    assert valid, f"Eigenvalue {error} not found in Pauli spectrum"


# ── _coefficients (JAX-pure) ───────────────────────────────────────────────


@pytest.mark.parametrize("hamil_random", [(4, 4)], indirect=True)
def test_coefficients_jax_pure(hamil_random):
    """_coefficients is JIT-compiled and JAX-differentiable; gradients are finite."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    tz = hamil_random.to_ElectronicStructureTZ()

    def loss(h1e, h2e):
        c = ElectronicStructureTZ._coefficients(h1e, h2e)
        return (
            jnp.sum(jnp.abs(c["T"]))
            + jnp.sum(jnp.abs(c["TT_opp"]))
            + jnp.sum(jnp.abs(c["TZ_same"]))
            + jnp.sum(jnp.abs(c["ZZ_opp"]))
        )

    val_and_grad = jax.value_and_grad(loss, argnums=(0, 1))
    val, (g1, g2) = val_and_grad(jnp.array(tz.h1e), jnp.array(tz.h2e))
    assert jnp.isfinite(val)
    assert g1.shape == tz.h1e.shape
    assert g2.shape == tz.h2e.shape
    assert bool(jnp.all(jnp.isfinite(g1)))
    assert bool(jnp.all(jnp.isfinite(g2)))


@pytest.mark.parametrize("hamil_random", [(4, 4)], indirect=True)
def test_one_norm_invariant_under_jw(hamil_random):
    """one_norm is permutation-invariant: applying any JW perm preserves Σ |c_α|."""
    import jax
    jax.config.update("jax_enable_x64", True)

    tz = hamil_random.to_ElectronicStructureTZ()
    rng = np.random.default_rng(7)
    perm = rng.permutation(tz.num_orb)
    tz_p = tz.apply_jw_ordering(perm)
    assert abs(tz.one_norm() - tz_p.one_norm()) < 1e-10


# ── jw_cost ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("hamil_random", [(4, 4)], indirect=True)
def test_jw_cost_matches_manual_sum(hamil_random):
    """jw_cost reproduces a brute-force Σ |c_α| · gates_α(perm) loop."""
    tz = hamil_random.to_ElectronicStructureTZ()
    n = tz.num_orb
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    pos = np.empty(n, dtype=int)
    pos[perm] = np.arange(n)

    expected = 0.0
    expected += 2.0 * float(np.sum(np.abs(tz.coeff_Z))) * tz.z_circuit_cost()
    expected += 2.0 * tz.zz_circuit_cost() * float(np.sum(np.abs(tz.coeff_ZZ_same)))
    expected += tz.zz_circuit_cost() * float(np.sum(np.abs(tz.coeff_ZZ_opp)))
    for p in range(n):
        for q in range(n):
            expected += 2.0 * abs(float(tz.coeff_T[p, q])) * tz.t_circuit_cost(pos[p], pos[q])
            for r in range(n):
                expected += 2.0 * abs(float(tz.coeff_TZ_opp[p, q, r])) * tz.tz_opp_circuit_cost(pos[p], pos[q])
                expected += 2.0 * abs(float(tz.coeff_TZ_same[p, q, r])) * tz.tz_same_circuit_cost(pos[p], pos[q], pos[r])
                for s in range(n):
                    expected += abs(float(tz.coeff_TT_opp[p, q, r, s])) * tz.tt_opp_circuit_cost(pos[p], pos[q], pos[r], pos[s])
                    expected += 2.0 * abs(float(tz.coeff_TT_same[p, q, r, s])) * tz.tt_same_circuit_cost(pos[p], pos[q], pos[r], pos[s])

    assert abs(tz.jw_cost(perm) - expected) < 1e-9


# ── apply_jw_ordering ──────────────────────────────────────────────────────


@pytest.mark.parametrize("hamil_random", [(4, 4)], indirect=True)
def test_apply_jw_ordering_preserves_pauli_spectrum(hamil_random):
    """Permuting orbital indices on the TZ side gives an equivalent Pauli
    Hamiltonian — the FCI spectrum stays a subset of the Pauli spectrum."""
    elstruc = hamil_random
    tz = elstruc.to_ElectronicStructureTZ()
    rng = np.random.default_rng(3)
    perm = rng.permutation(tz.num_orb)
    tz_p = tz.apply_jw_ordering(perm)
    pauli_hamil = tz_p.to_pauli_hamiltonian()
    valid, error = _f2q_valid(elstruc=elstruc, pauli_hamil=pauli_hamil)
    assert valid, f"Eigenvalue {error} not found in permuted Pauli spectrum"


@pytest.mark.parametrize("hamil_random", [(4, 4)], indirect=True)
def test_apply_jw_ordering_inverse_round_trip(hamil_random):
    """apply_jw_ordering(perm) followed by apply_jw_ordering(perm^-1) recovers
    the original h1e, h2e."""
    tz = hamil_random.to_ElectronicStructureTZ()
    rng = np.random.default_rng(4)
    perm = rng.permutation(tz.num_orb)
    inv = np.argsort(perm)
    tz_back = tz.apply_jw_ordering(perm).apply_jw_ordering(inv)
    assert np.allclose(np.asarray(tz_back.h1e), tz.h1e)
    assert np.allclose(np.asarray(tz_back.h2e), tz.h2e)


@pytest.mark.parametrize("hamil_random", [(4, 4)], indirect=True)
def test_apply_jw_ordering_inplace(hamil_random):
    """inplace=True rebuilds attributes; identity perm leaves coefficients unchanged."""
    tz = hamil_random.to_ElectronicStructureTZ()
    coeff_T_before = np.asarray(tz.coeff_T).copy()
    rv = tz.apply_jw_ordering(np.arange(tz.num_orb), inplace=True)
    assert rv is None
    assert np.allclose(np.asarray(tz.coeff_T), coeff_T_before)


# ── optimize_jw_ordering ───────────────────────────────────────────────────


@pytest.mark.parametrize("hamil_random", [(4, 4)], indirect=True)
def test_optimize_jw_ordering_no_worse_than_identity(hamil_random):
    """The returned permutation has jw_cost ≤ jw_cost(identity)."""
    tz = hamil_random.to_ElectronicStructureTZ()
    n = tz.num_orb
    perm = tz.optimize_jw_ordering()
    assert sorted(perm.tolist()) == list(range(n))  # is a permutation
    assert tz.jw_cost(perm) <= tz.jw_cost(np.arange(n)) + 1e-9
