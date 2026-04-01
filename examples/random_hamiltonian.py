"""
champy example: random electronic-structure Hamiltonian
========================================================
Demonstrates the main workflow:
  ElectronicStructure → MajoranaPair → Jordan-Wigner (PauliHamiltonian)
"""

import numpy as np
from champy.ElectronicStructure import ElectronicStructure
from champy.MajoranaPair import (
    MajoranaPair,
)  # noqa: F401 (imported via ElectronicStructure)

# ── 1. Build a random ElectronicStructure ────────────────────────────────────
NUM_ORB = 4
NUM_ELEC = 4
rng = np.random.default_rng(42)

h1e = rng.standard_normal((NUM_ORB, NUM_ORB))
h1e = h1e + h1e.T  # symmetrise

h2e = rng.standard_normal((NUM_ORB,) * 4)
h2e += np.einsum("pqrs->qprs", h2e)  # chemist symmetries
h2e += np.einsum("pqrs->pqsr", h2e)
h2e += np.einsum("pqrs->rspq", h2e)

elstruc = ElectronicStructure(h0=0.0, h1e=h1e, h2e=h2e, num_elec=NUM_ELEC)

print(f"Spatial orbitals : {elstruc.num_orb}")
print(f"Electrons        : {elstruc.num_elec}")
print(f"Fock-space dim   : {elstruc.dimension}")
print(f"Nuclear constant : {elstruc.constant:.6f}")

# ── 2. Ground-state energy (FCI via PySCF) ───────────────────────────────────
e_gs = elstruc.ground_state_energy()
print(f"\nFCI ground-state energy : {e_gs:.6f}")

# ── 3. Convert to MajoranaPair Hamiltonian ───────────────────────────────────
majorana = elstruc.to_MajoranaPair()

print(f"\nMajorana constant  : {majorana.constant:.6f}")
print(f"f1e shape          : {majorana.f1e.shape}")
print(f"f2e (same-spin) shape : {majorana.f2e_diffopp_samespin.shape}")

# ── 4. Commutation graph ─────────────────────────────────────────────────────
adj = majorana.commutation_graph()
num_ops = 2 * NUM_ORB**2
print(f"\nCommutation graph  : {num_ops}×{num_ops} adjacency matrix")
print(f"Non-zero entries   : {adj.sum()}")

# ── 5. Operator weights ───────────────────────────────────────────────────────
weights = majorana.majoranapair_weights()
print(f"\nMajorana-pair weights shape : {weights.shape}  (n×n×2 for ↑/↓)")
print(f"Max weight (↑ sector)       : {weights[:,:,0].max():.6f}")

# ── 6. Jordan-Wigner mapping → PauliHamiltonian ──────────────────────────────
pauli_hamil = majorana.jordan_wigner()
print(f"\nPauli Hamiltonian")
num_qubits = int(np.log2(pauli_hamil.dimension))
print(f"  Qubits : {num_qubits}")
print(f"  Terms  : {len(pauli_hamil.paulis)}")
print(f"  Constant : {pauli_hamil.constant:.6f}")

# ── 7. Spectral check: FCI eigenvalues ⊆ Pauli eigenvalues ──────────────────
spec_fci = (
    np.sort(np.real(np.linalg.eigvals(elstruc.to_sparse_matrix()))) + elstruc.constant
)
spec_pauli = (
    np.sort(np.real(np.linalg.eigvals(pauli_hamil.to_sparse_matrix().toarray())))
    + pauli_hamil.constant
)

print(f"\nLowest 4 FCI eigenvalues   : {spec_fci[:4]}")
print(f"Lowest 4 Pauli eigenvalues : {spec_pauli[:4]}")


# ── 8. Plots ─────────────────────────────────────────────────────────────────
majorana.plot_orbital_graph()
