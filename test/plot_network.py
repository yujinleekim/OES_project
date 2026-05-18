"""
plot_network.py
────────────────
Plot a map of the Wasterkingen grid showing buses, lines and device
locations (generators, PV, BESS and controllable loads) with their
name/value annotations.

Usage: run this script from the repository root. Saves an image to
`results/network_map.png` by default.
"""

from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

try:
    import networkx as nx
except Exception:
    nx = None

from grid_model import WasterkingenGrid


def _get_bus_positions(net):
    # Prefer explicit geodata if available
    if "x" in net.bus.columns and "y" in net.bus.columns:
        pos = {int(idx): (float(row.x), float(row.y))
               for idx, row in net.bus.iterrows()}
        return pos

    # Some pandapower files store geodata in bus_geodata or similar
    for colx, coly in [("geodata_x", "geodata_y"), ("coords_x", "coords_y")]:
        if colx in net.bus.columns and coly in net.bus.columns:
            pos = {int(idx): (float(row[colx]), float(row[coly]))
                   for idx, row in net.bus.iterrows()}
            return pos

    # If networkx is available, use a spring layout on the electrical graph
    nodes = list(net.bus.index)
    if nx is not None:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        for _, row in net.line.iterrows():
            u, v = int(row.from_bus), int(row.to_bus)
            if u in nodes and v in nodes:
                G.add_edge(u, v)
        p = nx.spring_layout(G, seed=42)
        # spring_layout gives positions keyed by node (same indices)
        pos = {int(k): (float(v[0]), float(v[1])) for k, v in p.items()}
        return pos

    # Fallback: place buses on a circle
    n = len(nodes)
    pos = {}
    for i, idx in enumerate(nodes):
        theta = 2 * math.pi * i / max(n, 1)
        pos[int(idx)] = (math.cos(theta), math.sin(theta))
    return pos


def plot_network(save_path: Path = None, show: bool = False):
    grid = WasterkingenGrid(data_dir=Path("data"))
    net = grid.net

    pos = _get_bus_positions(net)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(9, 7))

    # Draw lines (edges)
    for _, row in net.line.iterrows():
        u, v = int(row.from_bus), int(row.to_bus)
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], color="#888888", linewidth=1.2, zorder=1)

    # Draw buses
    bus_x = []
    bus_y = []
    bus_nodes = []
    for idx, (x, y) in pos.items():
        bus_x.append(x)
        bus_y.append(y)
        bus_nodes.append(idx)
    ax.scatter(bus_x, bus_y, s=40, color="#cccccc", edgecolors="#444444", zorder=2)

    # Annotate bus indices
    for idx, (x, y) in pos.items():
        ax.text(x, y - 0.02, str(idx), fontsize=7, ha="center", va="top", zorder=5)

    # Plot devices
    bess_list = grid.get_bess_devices()
    pv_list = grid.get_pv_devices()
    load_list = grid.get_controllable_loads()

    # Generators / external grid / sgens
    gens = []
    if hasattr(net, "ext_grid") and len(net.ext_grid) > 0:
        for _, r in net.ext_grid.iterrows():
            gens.append({"bus": int(r.bus), "label": "PCC"})
    if hasattr(net, "sgen") and len(net.sgen) > 0:
        for _, r in net.sgen.iterrows():
            gens.append({"bus": int(r.bus), "p_mw": float(r.get("p_mw", 0.0)),
                         "label": r.get("name", "Gen")})

    # Plot generator markers
    for g in gens:
        b = g["bus"]
        if b in pos:
            x, y = pos[b]
            ax.scatter([x], [y], marker="*", color="#2b83ba", s=150, zorder=6)
            lbl = g.get("label", "Gen")
            val = g.get("p_mw")
            if val is not None and not math.isnan(val):
                txt = f"{lbl}\n{val:.3f} MW"
            else:
                txt = f"{lbl}"
            ax.text(x, y + 0.03, txt, fontsize=8, ha="center", zorder=7)

    # Plot PV
    for pv in pv_list:
        try:
            b = int(pv.bus)
            x, y = pos[b]
            s = getattr(pv, "S_max_MW", getattr(pv, "p_max_mw", None))
            ax.scatter([x], [y], marker="^", color="#2ca25f", s=100, zorder=6)
            lbl = f"PV\n{s:.3f} MW" if s is not None else "PV"
            ax.text(x + 0.01, y + 0.01, lbl, fontsize=8, zorder=7)
        except Exception:
            continue

    # Plot BESS
    for bess in bess_list:
        try:
            b = int(bess.bus)
            x, y = pos[b]
            s = getattr(bess, "S_max_MW", None)
            e = getattr(bess, "E_cap_MWh", None)
            ax.scatter([x], [y], marker="s", color="#fdae61", s=100, zorder=6)
            parts = []
            if s is not None:
                parts.append(f"P:{s:.3f} MW")
            if e is not None:
                parts.append(f"E:{e:.3f} MWh")
            lbl = "BESS\n" + ", ".join(parts) if parts else "BESS"
            ax.text(x + 0.01, y - 0.01, lbl, fontsize=8, zorder=7)
        except Exception:
            continue

    # Plot controllable loads
    for ld in load_list:
        try:
            b = int(ld.bus)
            x, y = pos[b]
            p = getattr(ld, "P_max_MW", None)
            ax.scatter([x], [y], marker="v", color="#de2d26", s=90, zorder=6)
            lbl = f"Load\n{p:.3f} MW" if p is not None else "Load"
            ax.text(x - 0.02, y - 0.01, lbl, fontsize=8, ha="right", zorder=7)
        except Exception:
            continue

    ax.set_title("Wasterkingen Grid: buses, lines and devices")
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Wasterkingen network map")
    parser.add_argument("--out", type=Path, default=Path("results/network_map.png"),
                        help="Path to save the network map image")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    args = parser.parse_args()

    plot_network(save_path=args.out, show=args.show)


if __name__ == "__main__":
    main()
