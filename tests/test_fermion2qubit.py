import pytest

from champy.fermion2qubit import h2e_convention

import numpy as np

h2e_chemist = np.random.rand(8, 8, 8, 8)
h2e_chemist += np.einsum("pqrs -> qprs", h2e_chemist)
h2e_chemist += np.einsum("pqrs -> pqsr", h2e_chemist)
h2e_chemist += np.einsum("pqrs -> rspq", h2e_chemist)

h2e_physicist = np.random.rand(8, 8, 8, 8)
h2e_physicist += np.einsum("pqrs -> sqrp", h2e_physicist)
h2e_physicist += np.einsum("pqrs -> prqs", h2e_physicist)
h2e_physicist += np.einsum("pqrs -> qpsr", h2e_physicist)


@pytest.mark.parametrize(
    "h2e, convention", [(h2e_chemist, "chemist"), (h2e_physicist, "physicist")]
)
def test_check_h2e_convention(h2e, convention):
    assert h2e_convention(h2e) == convention
