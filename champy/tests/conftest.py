import pytest
from pyscf import gto, scf
from pyscf.mcscf import CASCI
from pyscf.ao2mo import restore
import numpy as np
from champy.ElectronicStructure import ElectronicStructure


def _rhf_h2o():
    # geometry from literature, Koridon '21 PRR
    xyz_H2O = """
    O 0. 0. 0.
    H 0.757 0.586 0.
    H -.757 0.586 0.
    """
    mol = gto.M(atom=xyz_H2O, basis="ccpvdz")
    hf = scf.RHF(mol).newton()
    hf.run()
    return hf


def _integrals_h2o(num_orb: int, num_elec: int):
    hf = _rhf_h2o()
    casci = CASCI(hf, num_orb, num_elec)
    h1e, h0 = casci.get_h1eff()
    h2e = restore("s1", casci.get_h2eff(), num_orb)
    return h0, h1e, h2e


def _integrals_random(num_orb, order):
    assert order in ["chemist", "physicist"]
    np.random.seed(4040)
    n = num_orb
    h1e = np.random.rand(n, n)
    h1e += np.einsum("pq -> qp", h1e)

    h2e = np.random.rand(n, n, n, n)
    if order == "chemist":
        h2e += np.einsum("pqrs -> qprs", h2e)
        h2e += np.einsum("pqrs -> pqsr", h2e)
        h2e += np.einsum("pqrs -> rspq", h2e)
    else:
        h2e += np.einsum("pqrs -> sqrp", h2e)
        h2e += np.einsum("pqrs -> prqs", h2e)
        h2e += np.einsum("pqrs -> qpsr", h2e)
    return 0, h1e, h2e


@pytest.fixture(scope="session")
def rhf_h2o(request):
    return _rhf_h2o()


@pytest.fixture(scope="session")
def integrals_h2o(request):
    return _integrals_h2o(request.param, request.param)


@pytest.fixture(scope="session")
def integrals_random(request):
    return _integrals_random(request.param, "chemist")


@pytest.fixture(scope="session")
def integrals_random_physicist(request):
    return _integrals_random(request.param, "physicist")


@pytest.fixture(scope="session")
def hamil_h2o(request):
    num_orb = request.param[0]
    num_elec = request.param[1]
    h0, h1e, h2e = _integrals_h2o(num_orb, num_elec)
    return ElectronicStructure(h0, h1e, h2e, num_elec)


@pytest.fixture(scope="session")
def hamil_random(request):
    num_orb = request.param[0]
    num_elec = request.param[1]
    h0, h1e, h2e = _integrals_random(num_orb, "chemist")
    return ElectronicStructure(h0, h1e, h2e, num_elec)


@pytest.fixture(scope="session")
def hamil_random_pair(request):
    h0_1, h1e_1, h2e_1 = _integrals_random(request.param["num_orb1"], "chemist")
    h0_2, h1e_2, h2e_2 = _integrals_random(request.param["num_orb2"], "chemist")
    hamil_1 = ElectronicStructure(h0_1, h1e_1, h2e_1, request.param["num_elec1"])
    hamil_2 = ElectronicStructure(h0_2, h1e_2, h2e_2, request.param["num_elec2"])
    return hamil_1, hamil_2
