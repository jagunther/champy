"""
champy example: H2O / cc-pVDZ active space
===========================================
Demonstrates the main workflow:
  ElectronicStructure → MajoranaPair → Jordan-Wigner (PauliHamiltonian)
"""

import numpy as np
from pyscf import gto, scf
from champy.ElectronicStructure import ElectronicStructure

# ── 1. Build ElectronicStructure from H2O / cc-pVDZ ─────────────────────────
NUM_ORB  = 6
NUM_ELEC = 6

mol = gto.M(atom="""
O 0. 0. 0.
H 0.757 0.586 0.
H -.757 0.586 0.
""", basis="ccpvdz")
rhf = scf.RHF(mol).newton()
rhf.run()

elstruc = ElectronicStructure.from_pyscf(rhf, num_orb=NUM_ORB, num_elec=NUM_ELEC)

print(f"Spatial orbitals : {elstruc.num_orb}")
print(f"Electrons        : {elstruc.num_elec}")
print(f"Fock-space dim   : {elstruc.dimension}")
print(f"Nuclear constant : {elstruc.constant:.6f}")

# ── 2. Symmetry detection ────────────────────────────────────────────────────
print(f"\nOrb symmetries (before ordering) : {elstruc.orb_symmetries}")
elstruc.plot_orbital_interaction_graph()

elstruc.symmetry_ordering()
print(f"Orb symmetries (after ordering)  : {elstruc.orb_symmetries}")
elstruc.plot_orbital_interaction_graph()

# ── 3. 1-norm optimisation ───────────────────────────────────────────────────
import copy

spc_before = elstruc.sum_pauli_coeffs()
print(f"\nsum_pauli_coeffs before optimisation : {spc_before:.6f}")

es_orb = copy.deepcopy(elstruc)
result_orb = es_orb.optimize_1norm(optimize_orbitals=True, optimize_shift=False, seed=0)
print(f"sum_pauli_coeffs after orbital-only  : {es_orb.sum_pauli_coeffs():.6f}  (converged: {result_orb.success})")

es_shift = copy.deepcopy(elstruc)
result_shift = es_shift.optimize_1norm(optimize_orbitals=False, optimize_shift=True, seed=0)
print(f"sum_pauli_coeffs after shift-only    : {es_shift.sum_pauli_coeffs():.6f}  (converged: {result_shift.success})")

es_both = copy.deepcopy(elstruc)
result_both = es_both.optimize_1norm(optimize_orbitals=True, optimize_shift=True, seed=0)
print(f"sum_pauli_coeffs after combined      : {es_both.sum_pauli_coeffs():.6f}  (converged: {result_both.success})")

# ── 4. Ground-state energy (FCI via PySCF) ───────────────────────────────────
e_gs = elstruc.ground_state_energy()
print(f"\nFCI ground-state energy : {e_gs:.6f}")

# ── 5. Convert to MajoranaPair Hamiltonian ───────────────────────────────────
majorana = elstruc.to_MajoranaPair()

print(f"\nMajorana constant     : {majorana.constant:.6f}")
print(f"f1e shape             : {majorana.f1e.shape}")
print(f"f2e (same-spin) shape : {majorana.f2e_diffopp_samespin.shape}")

# ── 6. Commutation graph ─────────────────────────────────────────────────────
adj = majorana.commutation_graph()
num_ops = 2 * NUM_ORB**2
print(f"\nCommutation graph : {num_ops}×{num_ops} adjacency matrix")
print(f"Non-zero entries  : {adj.sum()}")

# ── 7. Operator weights ───────────────────────────────────────────────────────
weights = majorana.majoranapair_weights()
print(f"\nMajorana-pair weights shape : {weights.shape}  (n×n×2 for ↑/↓)")
print(f"Max weight (↑ sector)       : {weights[:,:,0].max():.6f}")

# ── 8. Jordan-Wigner mapping → PauliHamiltonian ──────────────────────────────
pauli_hamil = majorana.jordan_wigner()
num_qubits = int(np.log2(pauli_hamil.dimension))
print(f"\nPauli Hamiltonian")
print(f"  Qubits   : {num_qubits}")
print(f"  Terms    : {len(pauli_hamil.paulis)}")
print(f"  Constant : {pauli_hamil.constant:.6f}")

# ── 9. Spectral check: FCI eigenvalues ⊆ Pauli eigenvalues ──────────────────
import scipy.sparse.linalg

spec_fci = (
    np.sort(scipy.sparse.linalg.eigsh(
        scipy.sparse.csr_matrix(elstruc.to_sparse_matrix()), k=25, which="SA", return_eigenvectors=False
    )) + elstruc.constant
)
spec_pauli = (
    np.sort(scipy.sparse.linalg.eigsh(
        pauli_hamil.to_sparse_matrix(), k=25, which="SA", return_eigenvectors=False
    )) + pauli_hamil.constant
)
print(f"\nLowest 4 FCI eigenvalues   : {spec_fci[:4]}")
print(f"Lowest 4 Pauli eigenvalues : {spec_pauli[:4]}")

# ── 10. Optimized JW ordering ────────────────────────────────────────────────
default_perm = np.arange(NUM_ORB)
opt_perm = majorana.optimize_jw_ordering()

print(f"\nDefault JW cost  : {majorana.jw_cost(default_perm):.4f}")
print(f"Optimized JW cost: {majorana.jw_cost(opt_perm):.4f}")
print(f"Optimal ordering : {opt_perm}")

# ── 11. Plots ────────────────────────────────────────────────────────────────
majorana.plot_orbital_graph(optimize_jw=True)
