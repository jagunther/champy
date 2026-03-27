# Hamiltonian formats

### Electronic Structure Hamiltonian


The `ElectronicStructure` class is the natural starting point of most calculation. 
It describes an electronic structure Hamiltonian in second quantized form:
$$
H \;=\; E_{\text{nuc}}
\;+\; \sum_{pq}\sum_{\sigma} h_{pq}\; a^\dagger_{p\sigma}\, a_{q\sigma}
\;+\; \frac{1}{2}\sum_{pqrs}\sum_{\sigma\tau} g_{pqrs}\; a^\dagger_{p\sigma}\, a^\dagger_{r\tau}\, a_{s\tau}\, a_{q\sigma}
$$
where
- $p, q, r, s$ run over **spatial** molecular orbitals,
- $\sigma, \tau \in \{\uparrow, \downarrow\}$ are spin indices,
- $E_{\text{nuc}}$ is the nuclear repulsion energy (a scalar).
The one-electron and two-electron integrals are defined by
$$
h_{pq} \;=\; \langle p \,|\, \hat{h} \,|\, q \rangle
\;=\; \int \phi_p^*(\mathbf{r})\left[-\tfrac{1}{2}\nabla^2 - \sum_A \frac{Z_A}{|\mathbf{r}-\mathbf{R}_A|}\right]\phi_q(\mathbf{r})\;\mathrm{d}\mathbf{r}.
$$
and
$$
g_{pqrs} \;=\;
\int\!\!\int
\frac{\phi_p^*(\mathbf{r}_1)\,\phi_q(\mathbf{r}_1)\;\phi_r^*(\mathbf{r}_2)\,\phi_s(\mathbf{r}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|}
\;\mathrm{d}\mathbf{r}_1\,\mathrm{d}\mathbf{r}_2,
$$
respectively.
Creating an `ElectronicStructure` instance requires specifying $E_{\text{nuc}}$, the 1-electron integrals $h_{pq}$, the 2-electron integrals $g_{pqrs}$ and the number of electrons.


### Majorana Pair Hamiltonian
TODO


### Pauli Hamiltonian
TODO