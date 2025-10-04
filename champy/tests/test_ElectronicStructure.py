import pytest
from champy.ElectronicStructure import ElectronicStructure
import numpy as np


@pytest.mark.parametrize("integrals_h2o", [4], indirect=True)
def test_ground_state(integrals_h2o):
    h0, h1e, h2e = integrals_h2o
    hamil = ElectronicStructure(h0=h0, h1e=h1e, h2e=h2e, num_elec=4)
    e_gs = hamil.ground_state_energy()
    h_matrix = hamil.to_sparse_matrix()
    eigvals, eigvecs = np.linalg.eigh(h_matrix)

    # ground state energy == lowest eigenvalue of hamiltonian matrix
    assert abs(min(eigvals) + hamil.constant - e_gs) < 1e-10

    # ground state == eigenvector of lowest eigenvalue of hamiltonian matrix
    assert abs(1 - abs(np.dot(eigvecs[:, 0], hamil.ground_state()))) < 1e-10


@pytest.mark.parametrize(
    "hamil_random_pair",
    [{"num_orb1": 4, "num_orb2": 4, "num_elec1": 4, "num_elec2": 4}],
    indirect=True,
)
def test_compatible(hamil_random_pair):
    hamil1, hamil2 = hamil_random_pair
    assert hamil1._compatible(hamil1)
    assert hamil2._compatible(hamil1)


@pytest.mark.parametrize(
    "hamil_random_pair",
    [
        {"num_orb1": 4, "num_orb2": 4, "num_elec1": 4, "num_elec2": 5},
        {"num_orb1": 4, "num_orb2": 5, "num_elec1": 4, "num_elec2": 4},
    ],
    indirect=True,
)
def test_compatible(hamil_random_pair):
    hamil1, hamil2 = hamil_random_pair
    assert not hamil1._compatible(hamil2)


@pytest.mark.parametrize(
    "hamil_random_pair",
    [
        {"num_orb1": 4, "num_orb2": 4, "num_elec1": 4, "num_elec2": 4},
        {"num_orb1": 6, "num_orb2": 6, "num_elec1": 4, "num_elec2": 4},
    ],
    indirect=True,
)
def test_add(hamil_random_pair):
    hamil1, hamil2 = hamil_random_pair
    hamil_add = hamil1 + hamil2
    assert hamil1.num_orb == hamil_add.num_orb
    assert hamil1.num_elec == hamil_add.num_elec

    e_gs = hamil1.ground_state_energy()
    assert abs(2 * e_gs - hamil_add.ground_state_energy()) < 1e-10


@pytest.mark.parametrize(
    "hamil_random_pair",
    [
        {"num_orb1": 4, "num_orb2": 4, "num_elec1": 4, "num_elec2": 4},
        {"num_orb1": 6, "num_orb2": 6, "num_elec1": 4, "num_elec2": 4},
    ],
    indirect=True,
)
def test_sub(hamil_random_pair):
    hamil1, hamil2 = hamil_random_pair
    hamil_sub = hamil1 - hamil2
    assert hamil1.num_orb == hamil_sub.num_orb
    assert hamil1.num_elec == hamil_sub.num_elec
    assert abs(hamil_sub.ground_state_energy()) < 1e-10


@pytest.mark.parametrize(
    "hamil_random",
    [
        (4, 4),
        (6, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "scalar",
    [0, -1, 1, 0.1234],
)
def test_mul(hamil_random, scalar):
    hamil = hamil_random
    e = hamil.ground_state_energy()
    if scalar > 0:
        hamil = scalar * hamil
        assert abs(hamil.ground_state_energy() - e * scalar) < 1e-10
    else:
        hamil = scalar * hamil
        assert abs(hamil.max_energy() + e * scalar) < 1e-10


@pytest.mark.parametrize("integrals_h2o", [4], indirect=True)
def test_fcidump(integrals_h2o, tmpdir):
    file = tmpdir / "FCIDUMP_test"
    h0, h1e, h2e = integrals_h2o
    hamil = ElectronicStructure(h0=h0, h1e=h1e, h2e=h2e, num_elec=4)
    hamil.to_fcidump(file)
    assert hamil == ElectronicStructure.from_fcidump(file)


@pytest.mark.parametrize("integrals_h2o", [4], indirect=True)
def test_from_pyscf(integrals_h2o, rhf_h2o):
    h0, h1e, h2e = integrals_h2o
    rhf = rhf_h2o
    hamil = ElectronicStructure.from_pyscf(rhf, 4, 4)
    assert np.allclose(h0, hamil.constant)


@pytest.mark.parametrize(
    "hamil_random_pair",
    [
        {"num_orb1": 4, "num_orb2": 4, "num_elec1": 4, "num_elec2": 4},
        {"num_orb1": 6, "num_orb2": 6, "num_elec1": 6, "num_elec2": 6},
    ],
    indirect=True,
)
def test_eq(hamil_random_pair):
    hamil1, hamil2 = hamil_random_pair
    assert hamil1 == hamil2


@pytest.mark.parametrize(
    "hamil_random_pair",
    [
        {"num_orb1": 4, "num_orb2": 5, "num_elec1": 4, "num_elec2": 4},
        {"num_orb1": 6, "num_orb2": 5, "num_elec1": 6, "num_elec2": 6},
    ],
    indirect=True,
)
def test_neq(hamil_random_pair):
    hamil1, hamil2 = hamil_random_pair
    assert hamil1 != hamil2


def test_sum_pauli_coeffs(rhf_h2o):
    rhf = rhf_h2o
    hamil = ElectronicStructure.from_pyscf(rhf, num_orb=24, num_elec=10)
    ref_value = 717  # from Koridon 2021 PRR
    assert abs(hamil.sum_pauli_coeffs() - ref_value) < 0.1


def test_is_canonical_hf_basis(rhf_h2o):
    rhf = rhf_h2o
    hamil = ElectronicStructure.from_pyscf(rhf, num_orb=24, num_elec=10)
    assert hamil.is_canonical_hf_basis()


def test_hf_energy(rhf_h2o):
    rhf = rhf_h2o
    e_ref = rhf.e_tot

    # compute hf energy without frozen core
    hamil = ElectronicStructure.from_pyscf(rhf, num_orb=10, num_elec=10)
    assert abs(e_ref - hamil.hf_energy()) < 1e-10

    # compute hf energy from hamil with frozen core
    hamil = ElectronicStructure.from_pyscf(rhf, num_orb=4, num_elec=4)
    assert abs(e_ref - hamil.hf_energy()) < 1e-10


def test_hf_orbital_energies(rhf_h2o):
    rhf = rhf_h2o
    hamil = ElectronicStructure.from_pyscf(rhf, num_orb=24, num_elec=10)
    assert np.allclose(hamil.hf_orbital_energies(), rhf.mo_energy)


def test_hf_state(rhf_h2o):
    rhf = rhf_h2o
    hamil = ElectronicStructure.from_pyscf(rhf, num_orb=4, num_elec=4)
    hf_state = hamil.hf_state()
    hf_expval = hf_state.T @ hamil.to_sparse_matrix() @ hf_state
    e_hf = hamil.hf_energy()
    assert abs(e_hf - (hf_expval + hamil.constant)) < 1e-7
