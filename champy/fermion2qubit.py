import numpy as np


def spinorb_from_spatial(h1e: np.ndarray, h2e: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Create coefficient tensors of electronic structure Hamiltonian w.r.t. spin orbitals, starting from
    the spatial orbital coefficients.
    Method adapted from openfermion, modified to not truncate the terms

    :param h1e: one-electron coefficients
    :param h2e: two-electron coefficients in chemists' convention
    """
    n = 2 * h1e.shape[0]  # number of spin orbitals

    h1e_spinorb = np.zeros((n, n))
    h2e_spinorb = np.zeros((n, n, n, n))
    for p in range(n // 2):
        for q in range(n // 2):
            # Populate 1-body coefficients. Require p and q have same spin.
            h1e_spinorb[2 * p, 2 * q] = h1e[p, q]
            h1e_spinorb[2 * p + 1, 2 * q + 1] = h1e[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n // 2):
                for s in range(n // 2):
                    # Mixed spin
                    h2e_spinorb[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = h2e[p, q, r, s]
                    h2e_spinorb[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = h2e[p, q, r, s]

                    # Same spin
                    h2e_spinorb[2 * p, 2 * q, 2 * r, 2 * s] = h2e[p, q, r, s]
                    h2e_spinorb[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = h2e[
                        p, q, r, s
                    ]
    return h1e_spinorb, h2e_spinorb


def h2e_convention(h2e: np.ndarray) -> str:
    """
    Check whether ordering of two-electron integral tensor follows physicists' or chemists' convention

    :param h2e: two-electron integral tensor
    :return: "chemist" or "physist" or RuntimeError if neither
    """

    if np.allclose(h2e, np.einsum("pqrs -> qprs", h2e)):
        if np.allclose(h2e, np.einsum("pqrs -> pqsr", h2e)):
            if np.allclose(h2e, np.einsum("pqrs -> rspq", h2e)):
                return "chemist"
    elif np.allclose(h2e, np.einsum("pqrs -> sqrp", h2e)):
        if np.allclose(h2e, np.einsum("pqrs -> prqs", h2e)):
            if np.allclose(h2e, np.einsum("pqrs -> qpsr", h2e)):
                return "physicist"
    raise RuntimeError("h2e does not have proper symmetry!")
