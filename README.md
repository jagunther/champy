# Hamiltonian formats

### Electronic Structure Hamiltonian


The `ElectronicStructure` class is the natural starting point of most calculation. 
It describes an electronic structure Hamiltonian in second quantized form:
$$
H = E_{\text{nuc}}
+ \sum_{pq}\sum_{\sigma} h_{pq} a^\dagger_{p\sigma}\, a_{q\sigma}
+ \frac{1}{2}\sum_{pqrs}\sum_{\sigma\tau} g_{pqrs} a^\dagger_{p\sigma}\, a^\dagger_{r\tau}\, a_{s\tau}\, a_{q\sigma}
$$
where
- $p, q, r, s$ run over **spatial** molecular orbitals,
- $\sigma, \tau \in \{\uparrow, \downarrow\}$ are spin indices,
- $E_{\text{nuc}}$ is the nuclear repulsion energy (a scalar).
The one-electron and two-electron integrals are defined by
$$
h_{pq} = \langle p \,|\, \hat{h} \,|\, q \rangle
= \int \phi_p^*(\mathbf{r})\left[-\tfrac{1}{2}\nabla^2 - \sum_A \frac{Z_A}{|\mathbf{r}-\mathbf{R}_A|}\right]\phi_q(\mathbf{r})\mathrm{d}\mathbf{r}.
$$
and
$$
g_{pqrs} =
\int\!\!\int
\frac{\phi_p^*(\mathbf{r}_1)\,\phi_q(\mathbf{r}_1)\phi_r^*(\mathbf{r}_2)\,\phi_s(\mathbf{r}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|}
\mathrm{d}\mathbf{r}_1\,\mathrm{d}\mathbf{r}_2,
$$
respectively.
Creating an `ElectronicStructure` instance requires specifying $E_{\text{nuc}}$, the 1-electron integrals $h_{pq}$, the 2-electron integrals $g_{pqrs}$ and the number of electrons.


### Majorana Pair Hamiltonian

Majorana operators are defined in terms of creation and annihilation operators via
$$
\gamma_{p\sigma,0} = a_{p\sigma} + a_{p\sigma}^{\dag},\qquad \gamma_{p\sigma,1} = -i(a_{p\sigma} - a_{p\sigma}^{\dag}).
$$
We define a Majorana pair operator as $\Gamma_{pq,\sigma} = i\gamma_{p\sigma,0}\gamma_{q\sigma,1} = a_{p\sigma}^{\dag}a_{q\sigma} - a_{p\sigma} a_{q\sigma}^{\dag}$.
Two Majorana pair operators anticommute if $\sigma = \tau$ and either $p=r$ OR $q=s$, in all other cases they commute.
Additionally, $\Gamma_{pq,\sigma}^2 = I$ and $\Gamma_{pq,\sigma}^{\dag} = \Gamma_{pq,\sigma}$, hence a set of Majorana pair operators is structurally equivalent to a set of Pauli operators with the same commutation structure.

The `MajoranaPair` class implements the Hamiltonian
$$
H = f_0 + \sum_{i}\sum_{\sigma} f_{i} \Gamma_{i\sigma} + \sum_{ij}\sum_{\sigma\tau}f_{ij}\Gamma_{i\sigma}\Gamma_{j\tau}.
$$
where $i = (p,q), j=(r,s)$ may be seen as paired indices of underlying fermionic modes (use `np.reshape()` for example).
The Hamiltonian is fully defined by the coefficients $f_0, f_i, f_{ij}$ and the commutation structure of the Majorana pairs.

For most purposes, a Majorana pair Hamiltonian is constructed by mapping an electronic structure Hamiltonian to a Majorana pair Hamiltonian with:
$$
f_0 = \sum_p h_{pp} + \frac12\sum_{pr}(g_{pprr} - g_{prrp}) \\
f_i = f_{pq} = \frac12\left(h_{pq} + \sum_r g_{pqrr} - \frac12\sum_r g_{prrq} \right)\\
f_{ij} = f_{pqrs} = \frac18 g_{pqrs}.
$$
Note that the 2-electron part of the Majorana pair Hamiltonian above contains duplicate terms. 
A minimal representation in which every term is unique is
$$
    \sum_{ij\sigma\tau} f_{ij}\Gamma_{i\sigma}\Gamma_{j\tau} = \left(2\sum_i f_{ii}\right) I + 2\sum_i f_{ii}\Gamma_{i\uparrow}\Gamma_{i\downarrow} + 2\sum_{i<j:[i,j]=0}\sum_{\sigma} f_{ij} \Gamma_{i\sigma}\Gamma_{j\sigma} + 2\sum_{i<j}f_{ij}(\Gamma_{i\uparrow}\Gamma_{j\downarrow} + \Gamma_{i\downarrow}\Gamma_{j\uparrow}).
$$

### Pauli Hamiltonian
TODO