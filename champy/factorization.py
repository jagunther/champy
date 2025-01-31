import numpy as np
import scipy


def cholesky_pivoted(A: np.ndarray, tol=1e-12) -> np.ndarray:
    """
    A wrapper around lapack.dpstrf
    https://stackoverflow.com/a/77821449
    """
    n = A.shape[0]
    L, piv, rank, info = scipy.linalg.lapack.dpstrf(A, lower=1, tol=tol)

    # remove the garbage entries Lapack did not care about
    L = np.tril(L)
    L = L[:, :rank]

    # create the permutation matrix
    # maybe there's a more pythonic way to permute L?
    P = np.zeros((n, n))
    P[piv - 1, np.arange(n)] = 1
    L = P @ L
    return L


def h2e_cholesky(h2e: np.ndarray) -> np.ndarray:
    """
    Decomposes the 'matricised' Coulomb tensor into Cholesky vectors:

        h_pqrs = Σ_j L(j)_pq L(j)_rs

    Returns R matrices L(j) each of shape N x N, where R is the rank of the
    Coulomb tensor.
    """
    norb = h2e.shape[0]
    h2e = np.reshape(h2e, (norb**2, norb**2))
    cholesky_vecs = cholesky_pivoted(h2e)
    rank = cholesky_vecs.shape[1]
    h2e_factors = np.zeros((rank, norb, norb))
    for i in range(rank):
        h2e_factors[i, :, :] = np.reshape(cholesky_vecs[:, i], (norb, norb))
    return h2e_factors


def cholesky_vec_to_integrals(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Yields the coefficients h_pq, h_pqrs of the hamiltonian term corresponding
    to a Cholesky-vector L of the Coulomb tensor decomposition. The coefficients
    encode a Hamiltonian of the form

        H = Σ_pq,σ h_pq a_pσ^+ a_qσ  +  1/2 Σ_pqrs,στ h_pqrs a_pσ^+ a_rτ^+ a_sτ a_qσ ,

    which is how make_openfermion_QubitOperator() needs the integral to be specified.
    A 1-electron part is present because the Coulomb tensor decomposition works with a
    different ordering of the 2-electron interaction.
    """
    h2e = np.einsum("pq,rs -> pqrs", L, L)
    h1e = 1 / 2 * np.einsum("pr,rq", L, L)
    return h1e, h2e


def h1e_factorized(h1e: np.ndarray, h2e: np.ndarray) -> np.ndarray:
    """
    Yields the modified 1-electron coefficients of the hamiltonian when reordering
    the 2-electron term from

        H = Σ_pq,σ h_pq a_pσ^+ a_qσ  +  1/2 Σ_pqrs,στ h_pqrs a_pσ^+ a_rτ^+ a_sτ a_qσ
    to
        H = Σ_pq,σ h'_pq a_pσ^+ a_qσ  +  1/2 Σ_pqrs,στ h_pqrs a_pσ^+ a_qσ a_rτ^+ a_sτ .
    """
    h1e_fact = h1e - 1 / 2 * np.einsum("prrq->pq", h2e)
    return h1e_fact


def h2e_eigen(h2e: np.ndarray, tol=1e-12):
    """
    Compute the eigendecomposition of the 'matricised' 2-electron integral tensor

    :arg h2e: 2-electron integral tensor
    :arg tol: eigenvector is only kept if abs(eigenvalue) > tol
    :return: The eigenvalues and eigenvectors of the decomposition
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    norb = h2e.shape[0]
    h2e = np.reshape(h2e, (norb**2, norb**2))
    eigvals, eigvecs = np.linalg.eigh(h2e)
    eig_sorted = [
        (val, vec)
        for val, vec in sorted(zip(eigvals, eigvecs.T), key=lambda t: abs(t[0]))
        if abs(val) > tol
    ][::-1]
    eigvals = np.array([e[0] for e in eig_sorted])
    eigvecs = np.array([e[1] for e in eig_sorted])
    rank = len(eigvecs)
    factors = np.zeros((rank, norb, norb))
    for i in range(rank):
        factors[i, :, :] = np.reshape(eigvecs[i], (norb, norb))
    return eigvals, factors

