import numpy as np
from openfermion import (
    QubitOperator,
    InteractionOperator,
    get_fermion_operator,
    jordan_wigner,
    bravyi_kitaev,
    symmetry_conserving_bravyi_kitaev,
    parity_code,
    binary_code_transform,
)


def integrals_to_QubitOperator(h0, h1e, h2e, mapping, nelec: int = None):
    """

    :param h0:
    :param h1e:
    :param h2e:
    :param mapping:
    :param nelec:
    :return:
    """
    h1e_spinorb, h2e_spinorb = spinorb_from_spatial(h1e, h2e)
    h2e_spinorb = h2e_order(h2e_spinorb, "physicist")
    op = get_fermion_operator(InteractionOperator(h0, h1e_spinorb, 1 / 2 * h2e_spinorb))
    num_spinorb = 2 * h1e.shape[0]
    if mapping == "jw":
        qubit_operator = jordan_wigner(op)
    elif mapping == "bk":
        qubit_operator = bravyi_kitaev(op)
    elif mapping == "symm-bk":
        if num_spinorb <= 4:
            pass
            # raise ValueError("Symm-BK on 2 spatial orbitals gives wrong eigenvalues!")
            # TODO: Fix this
        if nelec < 0:
            raise ValueError("number of electrons must be positive integer!")
        qubit_operator = symmetry_conserving_bravyi_kitaev(op, num_spinorb, nelec)
    elif mapping == "parity":
        code = parity_code(num_spinorb)
        qubit_operator = binary_code_transform(op, code)
    else:
        raise ValueError(f"Mapping {mapping} not supported")
    return qubit_operator


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


def h2e_check_convention(h2e: np.ndarray) -> str:
    """
    Check whether ordering of two-electron integral tensor follows physicists' or chemists' convention

    :param h2e: two-electron integral tensor
    :return: "chemist" or "physist"
    :raises RuntimeError: if order is neither chemists' nor physicists'
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


def h2e_order(h2e: np.ndarray, convention: str) -> np.ndarray:
    """
    Orders the two-electron integral tensor according to the given convention

    :param h2e: two-electron integral tensor
    :param convention: either 'chemist' or 'physicist'
    :return: two-electron is specified order
    """
    curr_convention = h2e_check_convention(h2e)
    if convention == curr_convention:
        return h2e
    elif convention == "physicist":
        return np.asarray(h2e.transpose(0, 2, 3, 1), order="C")
    elif convention == "chemist":
        return np.asarray(h2e.transpose(0, 3, 1, 2), order="C")
    else:
        raise ValueError(f"{convention} is not a valid convention")
