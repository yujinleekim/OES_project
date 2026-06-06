from pathlib import Path
import argparse
import math
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

try:
    import networkx as nx
except Exception:
    nx = None

from grid_model import WasterkingenGrid


def _get_bus_positions(grid, root_directory):
    """
    Forces the extraction of real geographic coordinates.
    If the grid dataframes don't have them, it manually reads the underlying Excel sheet.
    """
    # 1. Check if the grid object has a hidden geodata attribute
    for attr in ["bus_geodata", "geodata_df", "geo_df"]:
        if hasattr(grid, attr):
            df = getattr(grid, attr)
            if df is not None and not df.empty and "x" in df.columns and "y" in df.columns:
                return {int(idx): (float(row.x), float(row.y)) for idx, row in df.iterrows()}

    # 2. Check if x and y are simply inside the main bus_df
    if hasattr(grid, "bus_df") and grid.bus_df is not None and not grid.bus_df.empty:
        df = grid.bus_df
        if "x" in df.columns and "y" in df.columns:
            return {int(idx): (float(row.x), float(row.y)) for idx, row in df.iterrows()}

    # 3. FORCE BYPASS: Manually open the Excel sheet to grab 'bus_geodata' (like your working script)
    excel_path = root_directory / "70-1_0_4_grid.xlsx"
    if excel_path.exists():
        try:
            xls = pd.read_excel(excel_path, sheet_name=None, index_col=0)
            if 'bus_geodata' in xls:
                geo_df = xls['bus_geodata']
                if "x" in geo_df.columns and "y" in geo_df.columns:
                    print("--> Success: Force-loaded real coordinates from bus_geodata Excel sheet!")
                    return {int(idx): (float(row.x), float(row.y)) for idx, row in geo_df.iterrows()}
        except Exception as e:
            print(f"Warning: Failed manual Excel read bypass: {e}")

    # 4. Fallback to networkx layout if everything else completely fails
    print("--> WARNING: Could not find geographic geodata. Falling back to artificial layout.")
    if hasattr(grid, "bus_df") and grid.bus_df is not None:
        nodes = list(grid.bus_df.index)
        if nx is not None and hasattr(grid, "line_df") and grid.line_df is not None:
            G = nx.Graph()
            G.add_nodes_from(nodes)
            for _, row in grid.line_df.iterrows():
                u_col = "from_bus" if "from_bus" in row else row.index[0]
                v_col = "to_bus" if "to_bus" in row else row.index[1]
                u, v = int(row[u_col]), int(row[v_col])
                if u in nodes and v in nodes:
                    G.add_edge(u, v)
            p = nx.spring_layout(G, seed=42)
            return {int(k): (float(v[0]), float(v[1])) for k, v in p.items()}

        # Final loop fallback
        n = len(nodes)
        pos = {}
        for i, idx in enumerate(nodes):
            theta = 2 * math.pi * i / max(n, 1)
            pos[int(idx)] = (math.cos(theta), math.sin(theta))
        return pos
        
    return {}


def plot_network(save_path: Path = None, show: bool = False, new_line: tuple = None, grid=None):
    root_directory = Path(__file__).resolve().parent.parent
    if grid is None:
        grid = WasterkingenGrid(data_dir=root_directory)
        grid._BESS_DATA, grid._PV_DATA, grid._LOAD_DATA = grid._make_device_data()

    # Pass root_directory into the position engine to enable the manual file reader bypass
    pos = _get_bus_positions(grid, root_directory)

    fig, ax = plt.subplots(figsize=(11, 9))

    # Draw lines (edges)
    if hasattr(grid, "line_df") and grid.line_df is not None:
        added_new_line_label = False
        for _, row in grid.line_df.iterrows():
            u_col = "from_bus" if "from_bus" in row else row.index[0]
            v_col = "to_bus" if "to_bus" in row else row.index[1]
            u, v = int(row[u_col]), int(row[v_col])
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                # Check if this is our highlighted new line
                is_new = False
                if new_line and ((u, v) == new_line or (v, u) == new_line):
                    is_new = True
                # Fallback: check if the dataframe explicitly holds a candidate
                elif "name" in row and isinstance(row["name"], str) and row["name"].startswith("Cand_"):
                    is_new = True

                if is_new:
                    # Draw thick red dashed line
                    label = "Optimal New Line" if not added_new_line_label else None
                    ax.plot([x0, x1], [y0, y1], color="#e41a1c", linewidth=3.5, linestyle="--", zorder=4, label=label)
                    added_new_line_label = True
                else:
                    # Draw standard grey line
                    ax.plot([x0, x1], [y0, y1], color="#333333", linewidth=1.5, zorder=1)
                    
    # If the new line wasn't in the dataframe but was passed manually, draw it directly
    if new_line and not added_new_line_label:
        u, v = new_line
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], color="#e41a1c", linewidth=3.5, linestyle="--", zorder=4, label="Optimal New Line")

    # Draw buses
    bus_x = []
    bus_y = []
    for idx, (x, y) in pos.items():
        bus_x.append(x)
        bus_y.append(y)
    ax.scatter(bus_x, bus_y, s=40, color="#cccccc", edgecolors="#444444", zorder=2)

    # Annotate bus indices
    for idx, (x, y) in pos.items():
        ax.annotate(
            text=str(idx),
            xy=(x, y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
            weight='bold',
            color='#555555',
            zorder=5
        )

    # Gather data lists natively from your grid asset classes
    bess_list = grid.get_bess_devices()
    pv_list = grid.get_pv_devices()
    load_list = grid.get_controllable_loads()

    # Parse external grid / generators
    gens = []
    if hasattr(grid, "ext_grid_df") and grid.ext_grid_df is not None and len(grid.ext_grid_df) > 0:
        for _, r in grid.ext_grid_df.iterrows():
            bus_val = r.get("bus", r.index[0] if isinstance(r.index[0], (int, np.integer)) else 0)
            gens.append({"bus": int(bus_val), "label": "PCC"})
            
    if hasattr(grid, "sgen_df") and grid.sgen_df is not None and len(grid.sgen_df) > 0:
        for _, r in grid.sgen_df.iterrows():
            gens.append({"bus": int(r.get("bus", 0)), "p_mw": float(r.get("p_mw", 0.0)),
                         "label": r.get("name", "Gen")})

    # Plot generator markers
    for g in gens:
        b = g["bus"]
        if b in pos:
            x, y = pos[b]
            ax.scatter([x], [y], marker="*", color="#2b83ba", s=180, zorder=6)
            lbl = g.get("label", "Gen")
            val = g.get("p_mw")
            if val is not None and not math.isnan(val):
                txt = f"{lbl}\n{val:.3f} MW"
            else:
                txt = f"{lbl}"
            ax.text(x, y + 0.005, txt, fontsize=8, ha="center", weight='bold', zorder=7)

    # Plot PV
    for pv in pv_list:
        try:
            b = int(pv.bus)
            if b in pos:
                x, y = pos[b]
                s = getattr(pv, "S_max_MW", getattr(pv, "p_max_mw", None))
                ax.scatter([x], [y], marker="^", color="#2ca25f", s=110, zorder=6)
                lbl = f"PV\n{s:.3f} MW" if s is not None else "PV"
                ax.text(x + 0.002, y + 0.002, lbl, fontsize=8, weight='bold', zorder=7)
        except Exception:
            continue

    # Plot BESS
    for bess in bess_list:
        try:
            b = int(bess.bus)
            if b in pos:
                x, y = pos[b]
                s = getattr(bess, "S_max_MW", None)
                e = getattr(bess, "E_cap_MWh", None)
                ax.scatter([x], [y], marker="s", color="#fdae61", s=110, zorder=6)
                parts = []
                if s is not None:
                    parts.append(f"P:{s:.3f} MW")
                if e is not None:
                    parts.append(f"E:{e:.3f} MWh")
                lbl = "BESS\n" + ", ".join(parts) if parts else "BESS"
                ax.text(x + 0.002, y - 0.004, lbl, fontsize=8, weight='bold', zorder=7)
        except Exception:
            continue

    # Plot controllable loads
    for ld in load_list:
        try:
            b = int(ld.bus)
            if b in pos:
                x, y = pos[b]
                p = getattr(ld, "P_max_MW", None)
                ax.scatter([x], [y], marker="v", color="#de2d26", s=100, zorder=6)
                lbl = f"Load\n{p:.3f} MW" if p is not None else "Load"
                ax.text(x - 0.004, y - 0.002, lbl, fontsize=8, ha="right", weight='bold', zorder=7)
        except Exception:
            continue

    ax.set_title("Wasterkingen Grid: Baseline & Optimal Upgrade", fontsize=12, weight='bold')
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Add Legend to explain the new line
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
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