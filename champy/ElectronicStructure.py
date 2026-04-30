from champy.Hamiltonian import Hamiltonian
from champy.MajoranaPair import MajoranaPair
from champy.rotation import liealgebra_to_rotation
from pyscf import fci
from pyscf.tools import fcidump
from pyscf.ao2mo import restore
from pyscf.mcscf import CASCI
import numpy as np
import scipy
import scipy.optimize
import scipy.sparse
import scipy.sparse.csgraph
import scipy.linalg


class ElectronicStructure(Hamiltonian):

    def __init__(self, h0: float, h1e: np.ndarray, h2e, num_elec: int):

        # assert h1e and h2e have consistent shapes
        assert len(h1e.shape) == 2
        assert len(h2e.shape) == 4
        assert len(list(set(h1e.shape))) == 1
        assert set(h1e.shape) == set(h2e.shape)

        self.h0 = h0
        self.h1e = h1e
        self.h2e = h2e
        self.num_elec = num_elec
        self.hf_state = self._init_hf_state()
        self.orb_symmetries = self._find_orb_symmetries()
        super().__init__()

    def _init_hf_state(self) -> np.ndarray | None:
        """Return occupation vector [spin-up | spin-down] of length 2*num_orb if the default
        occupation (first num_occ orbitals) is a valid canonical HF state, else None."""
        assert self.num_elec % 2 == 0
        num_occ = self.num_elec // 2
        n = self.num_orb
        default_occ = np.arange(num_occ)

        # compute Fock with default occupation
        fock = self.h1e.copy()
        fock += 2 * np.sum(self.h2e[:, :, default_occ, default_occ], axis=-1)
        fock -= np.sum(self.h2e[:, default_occ, default_occ, :], axis=1)

        # check Fock is diagonal
        f_offdiag = fock.copy()
        np.fill_diagonal(f_offdiag, 0)
        if not np.all(np.abs(f_offdiag) < 1e-6):
            return None

        # check first num_occ orbital energies are the lowest
        orb_energies = np.diag(fock)
        if not np.all(orb_energies[:num_occ] <= orb_energies[num_occ:].min()):
            return None

        occ_vec = np.zeros(2 * n, dtype=int)
        occ_vec[:num_occ] = 1
        occ_vec[n : n + num_occ] = 1
        return occ_vec

    def _compatible(self, other):
        return self.num_elec == other.num_elec and self.num_orb == other.num_orb

    def __add__(self, other):
        if self._compatible(other):
            return ElectronicStructure(
                self.h0 + other.h0,
                self.h1e + other.h1e,
                self.h2e + other.h2e,
                self.num_elec,
            )
        else:
            raise RuntimeError(
                "ElectronicStructure objects must have same numbers of orbitals and electrons!"
            )

    def __sub__(self, other):
        if self._compatible(other):
            return ElectronicStructure(
                self.h0 - other.h0,
                self.h1e - other.h1e,
                self.h2e - other.h2e,
                self.num_elec,
            )
        else:
            raise RuntimeError(
                "ElectronicStructure objects must have same numbers of orbitals and electrons!"
            )

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return ElectronicStructure(
                other * self.h0, other * self.h1e, other * self.h2e, self.num_elec
            )
        else:
            raise TypeError(f"Cannot multiply ElectronicStructure by {type(other)}!")

    __rmul__ = __mul__

    def __eq__(self, other):
        if self._compatible(other):
            if np.allclose(other.h0, self.h0):
                if np.allclose(other.h1e, self.h1e):
                    if np.allclose(other.h2e, self.h2e):
                        return True
        return False

    @property
    def constant(self) -> float:
        return self.h0

    @property
    def num_orb(self) -> int:
        return self.h1e.shape[0]

    @property
    def dimension(self) -> int:
        return int(scipy.special.binom(self.num_orb, self.num_elec // 2) ** 2)

    def orbital_interaction_graph(self, threshold: float = 1e-7) -> np.ndarray:
        """Returns |h1e_pq| + Σ_r |h2e_pqrr|, with entries below threshold set to zero."""
        conn = np.abs(self.h1e) + np.einsum("pqrr->pq", np.abs(self.h2e))
        conn[conn < threshold] = 0.0
        return conn

    def _find_orb_symmetries(self, threshold: float = 1e-7) -> np.ndarray:
        """Assign each MO a connected-component index based on h1e and h2e connectivity."""
        conn = self.orbital_interaction_graph(threshold)
        adj = scipy.sparse.csr_matrix((conn > 0).astype(np.int8))
        _, labels = scipy.sparse.csgraph.connected_components(adj, directed=False)
        return labels

    def plot_orbital_interaction_graph(self) -> None:
        """Display the orbital interaction graph as a heatmap."""
        import matplotlib.pyplot as plt

        data = self.orbital_interaction_graph()
        cmap = plt.colormaps["Blues"].copy()
        cmap.set_under("lightgrey")

        vmin = data[data > 0].min() if np.any(data > 0) else 1.0

        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=data.max())
        fig.colorbar(
            im, ax=ax, label=r"$|h_{pq}^{1e}| + \sum_{r}|h_{pqrr}^{2e}|$", extend="min"
        )
        ax.set_xlabel("MO index q")
        ax.set_ylabel("MO index p")
        ax.set_title("Orbital interaction graph")
        plt.show()

    def symmetry_ordering(self) -> None:
        """Reorder MOs in-place by orb_symmetries index, grouping each symmetry block."""
        idx = np.argsort(self.orb_symmetries, kind="stable")
        self.h1e = self.h1e[np.ix_(idx, idx)]
        self.h2e = self.h2e[np.ix_(idx, idx, idx, idx)]
        self.orb_symmetries = self.orb_symmetries[idx]
        if self.hf_state is not None:
            n = self.num_orb
            self.hf_state = np.concatenate(
                [self.hf_state[:n][idx], self.hf_state[n:][idx]]
            )

    def to_sparse_matrix(self) -> scipy.sparse.csr_matrix:
        """
        Does not contain constant
        """
        return fci.direct_spin1.pspace(
            self.h1e, self.h2e, self.num_orb, self.num_elec, np=self.dimension
        )[1]

    def ground_state_energy(self) -> float:
        # The spin1 method works generally, see github.com/pyscf/pyscf/blob/master/examples/fci/01-given_h1e_h2e.py
        cisolver = fci.direct_spin1.FCI()
        cisolver.max_cycle = 100
        cisolver.conv_tol = 1e-10
        e, _ = cisolver.kernel(self.h1e, self.h2e, self.num_orb, self.num_elec)
        return e + self.constant

    def max_energy(self) -> float:
        cisolver = fci.direct_spin1.FCI()
        cisolver.max_cycle = 100
        cisolver.conv_tol = 1e-10
        e, _ = cisolver.kernel(-self.h1e, -self.h2e, self.num_orb, self.num_elec)
        return e - self.constant

    def ground_state(self) -> np.ndarray:
        _, fcivec = fci.direct_spin1.kernel(
            self.h1e,
            self.h2e,
            self.num_orb,
            self.num_elec,
            nroots=3,
            max_space=50,
            max_cycle=200,
        )
        return fcivec[0].flatten()

    def onv_basis(self) -> np.ndarray:
        """
        The occupation number vector basis w.r.t. which to_sparse_matrix(),
        ground_state() and hf_state() are defined. Each element is a string
        of length 2 * num_orb with 1s and 0s. The first num_orb bits represent
        the spin-alpha spin-orbitals, ordered with increasing orbital index
        from right to left. The second num_orb bits represent the spin-beta
        spin-orbitals, ordered the same way.

        Example: For 4 spatial orbitals (0,1,2,3) and 4 electrons the state with
        doubly occupied orbitals 0 and 1 is '00110011'.
        """
        assert self.num_elec % 2 == 0
        dets = [
            str(bin(x))[2:].rjust(self.num_orb, "0")
            for x in fci.cistring.make_strings(
                list(range(self.num_orb)), self.num_elec // 2
            )
        ]
        return np.array([[(d_a + d_b) for d_b in dets] for d_a in dets]).flatten()

    def fock_operator(self) -> np.ndarray:
        """
        Construct the Fock operator from the h1e and h2e

        :return: fock operator, num_orb x num_orb np.array
        """
        if self.hf_state is None:
            raise RuntimeError("hf_state is not set — not in canonical HF basis")
        occ = np.where(self.hf_state[: self.num_orb] == 1)[0]
        fock_op = self.h1e.copy()
        fock_op += 2 * np.sum(self.h2e[:, :, occ, occ], axis=-1)
        fock_op -= np.sum(self.h2e[:, occ, occ, :], axis=1)
        return fock_op

    def hf_orbital_energies(self) -> np.ndarray:
        """
        If h1e and h2e are given in the canonical HF basis, returns the orbital energies.
        """
        if not self.is_canonical_hf_basis():
            raise RuntimeError("h1e, h2e not in canonical HF basis")
        return np.diag(self.fock_operator())

    def is_canonical_hf_basis(self) -> bool:
        """
        Checks whether the underlying basis of h1e, h2e is the canonical HF basis
        """
        f = self.fock_operator()
        np.fill_diagonal(f, 0)
        if not np.all(np.abs(f) < 1e-6):
            return False
        if not np.allclose(np.diag(f), sorted(np.diag(f))):
            return False
        return True

    def hf_energy(self) -> float:
        """
        Compute the HF energy of the system. Checks first whether underlying orbitals are canonical
        HF orbitals. Assumes that orbitals are ordered by orbital energy.

        :return: HF energy
        """
        if self.hf_state is None:
            raise RuntimeError("hf_state is not set — not in canonical HF basis")
        if not self.is_canonical_hf_basis():
            raise RuntimeError("Not in canonical orbital basis")
        occ = np.where(self.hf_state[: self.num_orb] == 1)[0]
        e = self.constant
        e += 2 * np.sum(np.diag(self.h1e)[occ])
        e += 2 * np.einsum("iijj->", self.h2e[np.ix_(occ, occ, occ, occ)])
        e -= np.einsum("ijji->", self.h2e[np.ix_(occ, occ, occ, occ)])
        return e

    def hf_state_fci(self) -> np.ndarray:
        """
        Returns the HF state as a FCI vector (one-hot in the determinant basis).
        """
        if self.hf_state is None:
            raise RuntimeError("hf_state is not set — not in canonical HF basis")
        num_occ = self.num_elec // 2
        occ_orbs = np.where(self.hf_state[: self.num_orb] == 1)[0]
        hf_bitstring = int(sum(1 << int(i) for i in occ_orbs))
        hf_det_idx = fci.cistring.str2addr(self.num_orb, num_occ, hf_bitstring)
        num_det = int(scipy.special.binom(self.num_orb, num_occ))
        hf_det = np.zeros((num_det, num_det))
        hf_det[hf_det_idx, hf_det_idx] = 1
        return hf_det.flatten()

    def hf_overlap(self) -> float:
        """
        Computes overlap between the HF state and the ground state
        """
        return np.abs(self.hf_state_fci().T @ self.ground_state()) ** 2

    @staticmethod
    def from_pyscf(rhf, num_orb, num_elec):
        """
        Create an ElectronicStructure object from a PySCF RHF object

        :param rhf: converged PySCF RHF object
        :param num_orb: int, number of active spatial orbitals
        :param num_elec: int, number of active electrons
        :return: ElectronicStructure object
        """
        casci = CASCI(rhf, num_orb, num_elec)
        h1e, h0 = casci.get_h1eff()
        h2e = restore("s1", casci.get_h2eff(), num_orb)
        return ElectronicStructure(h0=h0, h1e=h1e, h2e=h2e, num_elec=num_elec)

    @staticmethod
    def from_fcidump(file: str):
        """
        Create an ElectronicStructure object from an FCIDUMP file
        :param file: path to the FCIDUMP file
        :return: ElectronicStructure object
        """
        data = fcidump.read(file)
        h0 = data["ECORE"]
        h1e = data["H1"]
        h2e = data["H2"]
        norb = data["NORB"]
        nelec = data["NELEC"]
        h2e = restore("s1", h2e, norb)
        return ElectronicStructure(h0=h0, h1e=h1e, h2e=h2e, num_elec=nelec)

    def to_fcidump(self, file: str):
        """
        Write the ElectronicStructure object to an FCIDUMP file

        :param file: path to file
        """
        fcidump.from_integrals(
            filename=file,
            h1e=self.h1e,
            h2e=self.h2e,
            nmo=self.num_orb,
            nelec=self.num_elec,
            nuc=self.constant,
        )

    def to_MajoranaPair(self):
        return MajoranaPair(h0=self.h0, h1e=self.h1e, h2e=self.h2e)

    def to_ElectronicStructureTZ(self):
        from champy.ElectronicStructureTZ import ElectronicStructureTZ

        return ElectronicStructureTZ(h0=self.h0, h1e=self.h1e, h2e=self.h2e)

    def sum_pauli_coeffs(self) -> float:
        """
        Compute the sum of absolute values of Pauli coefficients after the Jordan-Wigner mapping.
        Uses jw_matrix() for the correct linear map from Majorana coefficients to Pauli coefficients.
        """
        majorana = self.to_MajoranaPair()
        _, _, M = majorana.jw_matrix()
        x = np.concatenate(
            [
                np.array(majorana.f1e).ravel(),
                np.array(majorana.f2e_diffopp_samespin).ravel(),
                np.array(majorana.f2e_diffop_diffspin).ravel(),
                np.array(majorana.f2e_sameop_diffspin).ravel(),
            ]
        )
        return float(np.sum(np.abs(M @ x)))

    def shift_number_op(self, shift1e: float, shift2e: np.ndarray, inplace=False):
        assert shift2e.shape == (self.num_orb, self.num_orb)

        num_op = np.identity(self.num_orb)
        h0_shift = self.h0 - shift1e * self.num_elec
        h1e_shift = self.h1e + shift1e * num_op + (1 - self.num_elec) * shift2e
        h2e_shift = (
            self.h2e
            + np.einsum("pq,rs->pqrs", num_op, shift2e)
            + np.einsum("pq,rs->pqrs", shift2e, num_op)
        )

        if inplace:
            self.h0 = h0_shift
            self.h1e = h1e_shift
            self.h2e = h2e_shift
        else:
            return h0_shift, h1e_shift, h2e_shift

    def rotate_orbitals(self, rotation: np.ndarray, inplace=False):
        """
        Performs orbital rotation on the hamiltonian coefficients. The rotation o is either
        given directly as an element of O(n). Or it is given as a Lie-algebra element kappa in so(n),
        s.t. o = exp(kappa) and parameterized by the lower triangular entries x_kappa
        via np.tril_indices(norb, -1).

        :arg rotation: Rotation, either element of O(n) or so(n)
        :arg inplace: True for replacing integrals of self, False for returning rotated integrals
        :return: rotated Hamiltonian, integrals (default) or factorized format (if input is factorized)
        :rtype: typle[float, np.ndarray, np.ndarray]
        """

        if rotation.shape == (self.num_orb, self.num_orb):
            o = rotation
        elif rotation.shape == (self.num_orb * (self.num_orb - 1) // 2,):
            x_kappa = rotation
            o = liealgebra_to_rotation(self.num_orb, x_kappa)
        else:
            raise ValueError("rotation does not fit any format")

        # integrals h1e, h2e
        h1e_rot = np.einsum("pq,pr,qs->rs", self.h1e, o, o, optimize="optimal")
        h2e_rot = np.einsum("pqrs,pt->tqrs", self.h2e, o, optimize="optimal")
        h2e_rot = np.einsum("tqrs,qu->turs", h2e_rot, o, optimize="optimal")
        h2e_rot = np.einsum("turs,rv->tuvs", h2e_rot, o, optimize="optimal")
        h2e_rot = np.einsum("tuvs,sw->tuvw", h2e_rot, o, optimize="optimal")
        if inplace:
            self.h1e = h1e_rot
            self.h2e = h2e_rot
            self.hf_state = None
        else:
            return self.h0, h1e_rot, h2e_rot

    def optimize(
        self,
        objective_fn,
        optimize_orbitals: bool = True,
        optimize_shift: bool = True,
        method: str = "L-BFGS-B",
        perturbation: float = 1e-2,
        seed: int = None,
        inplace: bool = True,
        basinhopping: bool = False,
        basinhopping_kwargs: dict = None,
    ) -> scipy.optimize.OptimizeResult:
        """Minimize a cost function over orbital rotations and/or number-operator shifts.

        Both orbital rotations and shift2e are block-diagonal with respect to orb_symmetries:
        orbitals from different symmetry sectors are not mixed.

        Parameters
        ----------
        objective_fn : callable(h1e, h2e) -> float
            Cost function to minimize.
        optimize_orbitals : bool
        optimize_shift : bool
        method : str
            Local minimizer method passed to scipy.optimize.minimize, default 'L-BFGS-B'.
        perturbation : float
            Std of the Gaussian initial perturbation applied to all parameters, default 1e-2.
        seed : int or None
            Random seed for reproducibility.
        inplace : bool
            If True (default), apply the optimal parameters to h0, h1e, h2e in-place.
            If False, only return the OptimizeResult without modifying the instance.
        basinhopping : bool
            If True, wrap the local minimizer with scipy.optimize.basinhopping for
            global optimization. Default False.
        basinhopping_kwargs : dict or None
            Extra keyword arguments forwarded to scipy.optimize.basinhopping
            (e.g. niter, T, stepsize). Ignored when basinhopping=False.

        Returns
        -------
        scipy.optimize.OptimizeResult
        """
        if not optimize_orbitals and not optimize_shift:
            raise ValueError(
                "At least one of optimize_orbitals or optimize_shift must be True"
            )

        n = self.num_orb

        # block structure from orb_symmetries
        unique_labels = np.unique(self.orb_symmetries)
        blocks = [np.where(self.orb_symmetries == lbl)[0] for lbl in unique_labels]

        # κ: antisymmetric block-diagonal generator of orbital rotations O = exp(κ)
        # free parameters are the lower-triangular entries within each symmetry block
        block_tril_kappa = [(idx, np.tril_indices(len(idx), k=-1)) for idx in blocks]
        n_kappa = sum(len(ri) for _, (ri, _) in block_tril_kappa)

        import jax
        import jax.numpy as jnp

        h1e_jax = jnp.array(self.h1e)
        h2e_jax = jnp.array(self.h2e)
        num_op_jax = jnp.eye(n)

        # JAX-compatible unpack functions: use .at[].set() instead of in-place assignment
        def _unpack_kappa(x_kappa):
            kappa = jnp.zeros((n, n))
            offset = 0
            for idx, (ri, ci) in block_tril_kappa:
                nb_params = len(ri)
                vals = x_kappa[offset : offset + nb_params]
                offset += nb_params
                nb = len(idx)
                block = (
                    jnp.zeros((nb, nb)).at[(ri, ci)].set(vals).at[(ci, ri)].set(-vals)
                )
                kappa = kappa.at[np.ix_(idx, idx)].set(block)
            return kappa

        # shift2e: symmetric block-diagonal matrix (ξ in BLISS notation)
        # free parameters are the upper-triangular entries within each symmetry block
        block_triu_shift2e = [(idx, np.triu_indices(len(idx))) for idx in blocks]
        n_shift2e = sum(len(ri) for _, (ri, _) in block_triu_shift2e)

        def _unpack_shift2e(x_shift2e):
            shift2e = jnp.zeros((n, n))
            offset = 0
            for idx, (ri, ci) in block_triu_shift2e:
                nb = len(idx)
                nb_params = nb * (nb + 1) // 2
                vals = x_shift2e[offset : offset + nb_params]
                offset += nb_params
                block = (
                    jnp.zeros((nb, nb)).at[(ri, ci)].set(vals).at[(ci, ri)].set(vals)
                )
                shift2e = shift2e.at[np.ix_(idx, idx)].set(block)
            return shift2e

        # initial parameter vector: all parameters start from a small random perturbation
        rng = np.random.default_rng(seed)
        x0_parts = []
        if optimize_orbitals:
            x0_parts.append(rng.standard_normal(n_kappa) * perturbation)
        if optimize_shift:
            # shift1e (scalar μ₁) + shift2e (block-diagonal ξ entries)
            x0_parts.append(rng.standard_normal(1 + n_shift2e) * perturbation)
        x0 = np.concatenate(x0_parts)

        def _objective(x):
            pos = 0
            h1e_cur, h2e_cur = h1e_jax, h2e_jax

            if optimize_orbitals:
                kappa = _unpack_kappa(x[pos : pos + n_kappa])
                pos += n_kappa
                o = jax.scipy.linalg.expm(kappa)
                h1e_cur = jnp.einsum("pq,pr,qs->rs", h1e_cur, o, o)
                h2e_cur = jnp.einsum("pqrs,pt->tqrs", h2e_cur, o)
                h2e_cur = jnp.einsum("tqrs,qu->turs", h2e_cur, o)
                h2e_cur = jnp.einsum("turs,rv->tuvs", h2e_cur, o)
                h2e_cur = jnp.einsum("tuvs,sw->tuvw", h2e_cur, o)

            if optimize_shift:
                shift1e = x[pos]
                shift2e = _unpack_shift2e(x[pos + 1 : pos + 1 + n_shift2e])
                pos += 1 + n_shift2e
                h1e_cur = h1e_cur + shift1e * num_op_jax + (1 - self.num_elec) * shift2e
                h2e_cur = (
                    h2e_cur
                    + jnp.einsum("pq,rs->pqrs", num_op_jax, shift2e)
                    + jnp.einsum("pq,rs->pqrs", shift2e, num_op_jax)
                )

            return objective_fn(h1e_cur, h2e_cur)

        _val_and_grad = jax.jit(jax.value_and_grad(_objective))

        def _scipy_objective(x):
            val, grad = _val_and_grad(jnp.array(x))
            return float(val), np.array(grad)

        minimizer_kwargs = {"method": method, "jac": True}
        if basinhopping:
            bh_kwargs = basinhopping_kwargs or {}
            result = scipy.optimize.basinhopping(
                _scipy_objective,
                x0,
                minimizer_kwargs=minimizer_kwargs,
                seed=seed,
                **bh_kwargs,
            )
        else:
            result = scipy.optimize.minimize(_scipy_objective, x0, **minimizer_kwargs)

        if optimize_orbitals:
            x_opt = np.array(result.x, dtype=float)
            kappa = np.array(_unpack_kappa(x_opt[:n_kappa]), dtype=float)
            o = scipy.linalg.expm(kappa)
            result["orbrot"] = o

        if inplace:
            x_opt = np.array(result.x, dtype=float)
            pos = 0
            if optimize_orbitals:
                self.rotate_orbitals(result["orbrot"], inplace=True)
                pos += n_kappa
            if optimize_shift:
                shift1e = float(x_opt[pos])
                shift2e = np.array(
                    _unpack_shift2e(x_opt[pos + 1 : pos + 1 + n_shift2e]), dtype=float
                )
                self.shift_number_op(shift1e, shift2e, inplace=True)

        return result

    def optimize_1norm(
        self,
        optimize_orbitals: bool = True,
        optimize_shift: bool = True,
        method: str = "L-BFGS-B",
        perturbation: float = 1e-2,
        seed: int = None,
        inplace: bool = True,
        basinhopping: bool = False,
        basinhopping_kwargs: dict = None,
    ) -> scipy.optimize.OptimizeResult:
        """Minimize the Pauli 1-norm over orbital rotations and/or number-operator shifts.

        M is precomputed once from jw_matrix() for the current orbital ordering and
        closed over in the objective. This makes the objective JAX-differentiable via
        _majorana_coeffs().
        """
        import jax.numpy as jnp

        _, _, M_sparse = self.to_MajoranaPair().jw_matrix()
        M = jnp.array(M_sparse.toarray().real)

        def _1norm(h1e, h2e):
            _, f1e, f2e_ss, f2e_ds, f2e_so = MajoranaPair._majorana_coeffs(
                0.0, h1e, h2e
            )
            x = jnp.concatenate(
                [f1e.ravel(), f2e_ss.ravel(), f2e_ds.ravel(), f2e_so.ravel()]
            )
            return jnp.sum(jnp.abs(M @ x))

        return self.optimize(
            _1norm,
            optimize_orbitals=optimize_orbitals,
            optimize_shift=optimize_shift,
            method=method,
            perturbation=perturbation,
            seed=seed,
            inplace=inplace,
            basinhopping=basinhopping,
            basinhopping_kwargs=basinhopping_kwargs,
        )
