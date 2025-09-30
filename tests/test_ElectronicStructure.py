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
    print(hamil1.h1e, hamil2.h1e)
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
def test_fcidump(integrals_h2o):
    h0, h1e, h2e = integrals_h2o
    hamil = ElectronicStructure(h0=h0, h1e=h1e, h2e=h2e, num_elec=4)
    hamil.to_fcidump("FCIDUMP_test")
    hamil_ = ElectronicStructure.from_fcidump("FCIDUMP_test")
