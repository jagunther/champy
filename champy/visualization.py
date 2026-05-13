"""Visualization helpers. Plotting routines for champy Hamiltonian objects.

Kept in a separate module so the core classes don't drag in matplotlib +
networkx as effective import-time dependencies.
"""

import numpy as np


def plot_orbital_interaction_graph(es) -> None:
    """Display the orbital interaction graph of an ElectronicStructure as a heatmap.

    Shows |h1e_pq| + Σ_r |h2e_pqrr| with a Blues colormap; entries below the
    interaction-graph threshold are rendered light grey via cmap.set_under.
    """
    import matplotlib.pyplot as plt

    data = es.orbital_interaction_graph()
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


def plot_orbital_graph_tz(tz, optimize_jw: bool = False) -> None:
    """Plot the orbital graph of an ElectronicStructureTZ using a spring layout,
    optionally with JW orderings overlaid as paths.

    Vertex weight: |coeff_Z[p]|. Edge weight: aggregated hopping-pair
    magnitude from tz._jw_pair_weights() (same matrix that seeds
    optimize_jw_ordering). Heavier edges pull nodes closer in the layout.

    :param tz: ElectronicStructureTZ instance.
    :param optimize_jw: if True, overlay the optimized JW ordering in
                        addition to the default identity ordering.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import matplotlib.colors as mcolors
    import matplotlib.colorbar as mcolorbar

    n = tz.num_orb
    w = tz._jw_pair_weights()
    diag_vals = np.abs(tz.coeff_Z)

    cmap = plt.colormaps["Blues"]
    nonzero = w[w > 0]
    vmin = nonzero.min() if nonzero.size > 0 else 1e-12
    vmax = max(w.max(), diag_vals.max(), vmin * 10)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    # Build graph with edge weights and compute layout
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for p in range(n):
        for q in range(p + 1, n):
            if w[p, q] > 1e-6 * w.max():
                G.add_edge(p, q, weight=w[p, q])
    pos = nx.spring_layout(G, weight="weight", seed=42)

    # JW orderings to display
    jw_orderings = [("JW default", np.arange(n), "red")]
    if optimize_jw:
        jw_orderings.append(("JW optimized", tz.optimize_jw_ordering(), "green"))

    n_cols = 1 + len(jw_orderings)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    fig.subplots_adjust(right=0.88)

    def _draw_vertices(ax):
        for p in range(n):
            circle = plt.Circle(
                pos[p],
                0.07,
                facecolor=(
                    cmap(norm(diag_vals[p])) if diag_vals[p] > 0 else "lightgrey"
                ),
                edgecolor="black",
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(circle)
            if diag_vals[p] > 0:
                r, g, b, _ = cmap(norm(diag_vals[p]))
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "white" if luminance < 0.5 else "black"
            else:
                text_color = "black"
            ax.text(
                *pos[p],
                str(p),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                zorder=4,
                color=text_color,
            )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.autoscale_view()

    # ── Left: orbital interaction graph ──────────────────────────────────
    ax = axes[0]
    for p, q in G.edges():
        xs = [pos[p][0], pos[q][0]]
        ys = [pos[p][1], pos[q][1]]
        ax.plot(xs, ys, color=cmap(norm(w[p, q])), lw=2, zorder=1)
    _draw_vertices(ax)
    ax.set_title(
        "Orbital graph\nvertex: $|c_{Z_p}|$,  edge: hopping-pair weight",
        fontsize=10,
    )

    # ── Right: one subplot per JW ordering ───────────────────────────────
    for ax, (title, perm, color) in zip(axes[1:], jw_orderings):
        cost = tz.jw_cost(perm)
        for i in range(n - 1):
            xs = [pos[perm[i]][0], pos[perm[i + 1]][0]]
            ys = [pos[perm[i]][1], pos[perm[i + 1]][1]]
            ax.plot(xs, ys, color=color, lw=2, zorder=1)
        _draw_vertices(ax)
        ax.set_title(f"{title}\ncost = {cost:.3f}", fontsize=10)

    # Shared colorbar
    cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    mcolorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    cax.set_title(r"$w$", fontsize=10)

    plt.show()


def plot_orbital_graph_majorana(majorana, optimize_jw: bool = False) -> None:
    """Plot the orbital graph for the spin-↑ sector of a MajoranaPair using a
    spring layout.

    Edge weights drive the spring forces: heavier edges pull nodes closer.
    Γ_pp → vertex p  (color proportional to weight)
    Γ_pq → undirected edge  (color proportional to weight)

    :param majorana: MajoranaPair instance.
    :param optimize_jw: if True, compute and display the optimal JW ordering
                        instead of the default 0,1,...,n-1.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import matplotlib.colors as mcolors
    import matplotlib.colorbar as mcolorbar

    n = majorana.num_orb
    w = majorana.majoranapair_weights()[:, :, 0]

    cmap = plt.colormaps["Blues"]
    diag_vals = w[np.arange(n), np.arange(n)]
    nonzero = w[w > 0]
    norm = mcolors.LogNorm(vmin=nonzero.min(), vmax=w.max())

    # Build graph with edge weights and compute layout
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for p in range(n):
        for q in range(p + 1, n):
            if w[p, q] > 1e-6 * w.max():
                G.add_edge(p, q, weight=w[p, q])
    pos = nx.spring_layout(G, weight="weight", seed=42)

    # JW orderings to display
    jw_orderings = [("JW default", np.arange(n), "red")]
    if optimize_jw:
        jw_orderings.append(
            ("JW optimized", majorana.optimize_jw_ordering(), "green")
        )

    n_cols = 1 + len(jw_orderings)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    fig.subplots_adjust(right=0.88)

    def _draw_vertices(ax):
        for p in range(n):
            circle = plt.Circle(
                pos[p],
                0.07,
                facecolor=cmap(norm(diag_vals[p])),
                edgecolor="black",
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(circle)
            r, g, b, _ = cmap(norm(diag_vals[p]))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(
                *pos[p],
                str(p),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                zorder=4,
                color="white" if luminance < 0.5 else "black",
            )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.autoscale_view()

    # ── Left: orbital interaction graph ──────────────────────────────────
    ax = axes[0]
    for p, q in G.edges():
        xs = [pos[p][0], pos[q][0]]
        ys = [pos[p][1], pos[q][1]]
        ax.plot(xs, ys, color=cmap(norm(w[p, q])), lw=2, zorder=1)
    _draw_vertices(ax)
    ax.set_title(
        "Orbital graph (spin-↑)\n"
        r"$\Gamma_{pp}$ → vertex,  $\Gamma_{pq}$ → edge",
        fontsize=10,
    )

    # ── Right: one subplot per JW ordering ───────────────────────────────
    for ax, (title, perm, color) in zip(axes[1:], jw_orderings):
        cost = majorana.jw_cost(perm)
        for i in range(n - 1):
            xs = [pos[perm[i]][0], pos[perm[i + 1]][0]]
            ys = [pos[perm[i]][1], pos[perm[i + 1]][1]]
            ax.plot(xs, ys, color=color, lw=2, zorder=1)
        _draw_vertices(ax)
        ax.set_title(f"{title}\ncost = {cost:.3f}", fontsize=10)

    # Shared colorbar
    cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    mcolorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    cax.set_title(r"$w$", fontsize=10)

    plt.show()
