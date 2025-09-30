import pytest
from champy.factorization import (
    cholesky_pivoted,
    h2e_cholesky,
    h2e_eigen,
    cholesky_vec_to_integrals,
)
import numpy as np


@pytest.mark.parametrize("n, m", [(5, 3), (10, 10)])
def test_lapack_dpstrf(n, m):
    A = np.random.rand(n, m)
    A = A @ A.T
    L = cholesky_pivoted(A)
    assert np.allclose(A, L @ L.T)


@pytest.mark.parametrize("integrals_h2o", [4], indirect=True)
def test_h2e_cholesky(integrals_h2o):
    h2e = integrals_h2o[2]
    h2e_factors = h2e_cholesky(h2e)
    for factor in h2e_factors:
        assert np.allclose(factor, factor.T)
    h2e_reconstructed = np.einsum("jpq,jrs -> pqrs", h2e_factors, h2e_factors)
    assert np.allclose(h2e, h2e_reconstructed)


@pytest.mark.parametrize("integrals_h2o", [4], indirect=True)
def test_h2e_eigen(integrals_h2o):
    h2e = integrals_h2o[2]
    eigvals, h2e_factors = h2e_eigen(h2e)
    for factor in h2e_factors:
        assert np.allclose(factor, factor.T)
    h2e_reconstructed = np.einsum(
        "j,jpq,jrs -> pqrs", eigvals, h2e_factors, h2e_factors
    )
    assert np.allclose(h2e, h2e_reconstructed)


@pytest.mark.parametrize("integrals_h2o", [4], indirect=True)
def test_integrals_df_factor(integrals_h2o):
    h2e = integrals_h2o[2]
    h2e_factors = h2e_cholesky(h2e)
    h2e_factors_integrals_h2e = [cholesky_vec_to_integrals(L) for L in h2e_factors]
    h1e_correction = np.sum([intg[0] for intg in h2e_factors_integrals_h2e], axis=0)
    h2e_reconstructed = np.sum([intg[1] for intg in h2e_factors_integrals_h2e], axis=0)
    assert np.allclose(h2e, h2e_reconstructed)
    assert np.allclose(h1e_correction, 1 / 2 * np.einsum("prrq->pq", h2e))
