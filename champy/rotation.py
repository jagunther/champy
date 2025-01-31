import numpy as np
import scipy.linalg

# exp: so(n) --> SO(n) is not surjective, not even on the identity component of SO(n),
# e.g.: -id(2) is in SO(n), but there is no X in so(n) s.t. exp(X) = -id(2)
# Therefore liealgebra -> rotation -> liealgebra is identity,
# but rotation -> liealgebra -> rotation is not.


def liealgebra_to_rotation(n: int, x_kappa: np.ndarray) -> np.ndarray:
    """
    The rotation o ∈ SO(n) is constructed as

        o = exp(kappa)   with  kappa = -kappa^T ∈ so(n)

    x_kappa defines the lower triangular entries of kappa.
    """
    assert len(x_kappa) == n * (n - 1) // 2
    assert x_kappa.dtype == float

    tril = np.tril_indices(n, -1)
    kappa = np.zeros((n, n), dtype=float)
    kappa[tril] = x_kappa
    kappa -= kappa.conj().T
    return scipy.linalg.expm(kappa)


def rotation_to_liealgebra(o: np.ndarray) -> np.ndarray:
    """
    Gives the lie algebra element kappa = -kappa^T ∈ so(n) such that

        log(o) = kappa

    Returns the lower triangular entries of kappa.
    """
    n = o.shape[0]
    eigvals, v = np.linalg.eig(o)
    eigphases = 1j * np.angle(eigvals)
    kappa = v @ np.diag(eigphases) @ v.conj().T
    indices = np.tril_indices(n, -1)
    return np.real(kappa[indices])
