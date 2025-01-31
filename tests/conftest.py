import pytest
from pyscf import gto, scf
from pyscf.mcscf import CASCI
from pyscf.ao2mo import restore


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

