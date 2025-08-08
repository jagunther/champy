import pytest
from pyscf import gto, scf
from pyscf.mcscf import CASCI
from pyscf.ao2mo import restore
import numpy as np


@pytest.fixture(scope="session")
def integrals_h2o(request):
    xyz_H2O = """
    O          0.00000        0.00000        0.11779
    H          0.00000        0.75545       -0.47116
    H          0.00000       -0.75545       -0.47116
    """
    n = request.param
    mol = gto.M(atom=xyz_H2O, basis="sto3g")
    hf = scf.RHF(mol).newton()
    hf.run()
    casci = CASCI(hf, n, n)
    h1e, h0 = casci.get_h1eff()
    h2e = restore("s1", casci.get_h2eff(), n)
    return h0, h1e, h2e


@pytest.fixture(scope="session")
def integrals_random(request):
    np.random.seed(4040)
    n = request.param
    h1e = np.random.rand(n, n)
    h1e += np.einsum("pq -> qp", h1e)

    h2e = np.random.rand(n, n, n, n)
    h2e += np.einsum("pqrs -> qprs", h2e)
    h2e += np.einsum("pqrs -> pqsr", h2e)
    h2e += np.einsum("pqrs -> rspq", h2e)

    return 0, h1e, h2e


@pytest.fixture(scope="session")
def integrals_random_physicist(request):
    np.random.seed(4040)
    n = request.param
    h1e = np.random.rand(n, n)
    h1e += np.einsum("pq -> qp", h1e)

    h2e = np.random.rand(n, n, n, n)
    h2e += np.einsum("pqrs -> sqrp", h2e)
    h2e += np.einsum("pqrs -> prqs", h2e)
    h2e += np.einsum("pqrs -> qpsr", h2e)

    return 0, h1e, h2e
