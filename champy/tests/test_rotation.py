import pytest

from champy.rotation import (
    liealgebra_to_rotation,
    rotation_to_liealgebra,
)
import numpy as np


@pytest.mark.parametrize("n", [4, 5, 6])
def test_liealgebra_to_rotation_consistency(n):
    for i in range(10):
        kappa = np.random.rand(n, n)
        kappa -= kappa.T
        x_kappa = kappa[np.tril_indices(n - 1)]
        o = liealgebra_to_rotation(n, x_kappa)
        x_kappa_ = rotation_to_liealgebra(o)
        assert np.allclose(x_kappa_, x_kappa)
