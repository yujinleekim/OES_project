"""
line_add.py
───────────
Find the optimal location between two unconnected vertices in WasterkingenGrid
to add an edge (line) such that the FFOR area is maximised.

Approach
────────
1. Pre-filter candidates to the 20 most electrically significant pairs using
   a fast proxy score (sum of |J_inv| sensitivity rows for both buses).
2. For each of the candidates, temporarily add the new line to the line_df,
   rebuild the Jacobian (Y-matrix only), and recompute the FFOR.
3. Score each candidate by the convex-hull area of the resulting FFOR polygon.
4. Report the winning pair and the area gain vs baseline.
"""

import copy
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import ConvexHull

from grid_model import WasterkingenGrid, _build_jacobian
from devices import BESSModel, PVModel, ControllableLoadModel
from ffor_optimizer import FFOROptimizer
from utils import plot_ffor


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ffor_area(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Convex-hull area of the FFOR boundary points.
    Returns 0.0 if fewer than 3 valid (non-NaN) points are available.
    """
    mask = ~(np.isnan(P) | np.isnan(Q))
    pts  = np.column_stack([P[mask], Q[mask]])
    if len(pts) < 3:
        return 0.0
    try:
        return ConvexHull(pts).volume   # 'volume' = area in 2-D
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Candidate-line helpers
# ─────────────────────────────────────────────────────────────────────────────

def _existing_edges(line_df: pd.DataFrame) -> set:
    """Return the set of unordered bus-index pairs that already have a line."""
    edges = set()
    for _, row in line_df.iterrows():
        u, v = int(row['from_bus']), int(row['to_bus'])
        edges.add((min(u, v), max(u, v)))
    return edges


def _candidate_pairs(bus_df: pd.DataFrame, line_df: pd.DataFrame,
                     max_hops: int = 3) -> list:
    """
    Return candidate (bus_i, bus_j) pairs for new line addition.
    Only considers pairs that are not already connected and are within max_hops.
    """
    try:
        import networkx as nx
    except ImportError:
        bus_indices = list(bus_df.index)
        existing    = _existing_edges(line_df)
        return [(u, v)
                for u, v in itertools.combinations(bus_indices, 2)
                if (min(u, v), max(u, v)) not in existing]

    G = nx.Graph()
    for _, row in line_df.iterrows():
        if row.get('in_service', True):
            G.add_edge(int(row['from_bus']), int(row['to_bus']))

    existing = _existing_edges(line_df)
    candidates = []
    for u in list(G.nodes()):
        lengths = nx.single_source_shortest_path_length(G, u, cutoff=max_hops)
        for v, hops in lengths.items():
            if v <= u:
                continue
            if hops < 2:
                continue
            if (min(u, v), max(u, v)) in existing:
                continue
            candidates.append((u, v))
    return candidates


def _score_candidates(candidates: list, J_inv: np.ndarray,
                      non_pcc_positions: list, top_n: int = 20) -> list:
    """
    Fast proxy score: for each candidate pair (i, j), estimate how much the
    new line will change the FFOR by looking at the L1 norm of the voltage-
    sensitivity rows for both buses.
    """
    n_pq   = len(non_pcc_positions)
    J_volt = J_inv[n_pq:]

    pos_map = {bus: p for p, bus in enumerate(non_pcc_positions)}

    scored = []
    for u, v in candidates:
        pu = pos_map.get(u, -1)
        pv = pos_map.get(v, -1)
        score = 0.0
        if pu >= 0:
            score += float(np.sum(np.abs(J_volt[pu])))
        if pv >= 0:
            score += float(np.sum(np.abs(J_volt[pv])))
        scored.append((score, u, v))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(u, v) for _, u, v in scored[:top_n]]


def _top_significant_buses(J_inv: np.ndarray,
                           non_pcc_positions: list,
                           top_n: int = 20) -> list:
    """Rank buses by the L1 norm of their voltage-sensitivity rows in J_inv."""
    n_pq   = len(non_pcc_positions)
    J_volt = J_inv[n_pq:]
    pos_map = {bus: p for p, bus in enumerate(non_pcc_positions)}

    scored = []
    for bus in non_pcc_positions:
        p = pos_map[bus]
        scored.append((float(np.sum(np.abs(J_volt[p]))), bus))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [bus for _, bus in scored[:top_n]]


def _median_line_params(line_df: pd.DataFrame) -> dict:
    """Extract representative per-km line parameters from the existing network."""
    return {
        "r_ohm_per_km": float(np.median(line_df['r_ohm_per_km'])),
        "x_ohm_per_km": float(np.median(line_df['x_ohm_per_km'])),
        "c_nf_per_km":  float(np.median(line_df['c_nf_per_km'])),
        "max_i_ka":     float(np.median(line_df['max_i_ka'])),
    }


def _add_candidate_line(line_df: pd.DataFrame, bus_i: int, bus_j: int,
                        length_km: float, line_params: dict):
    """Add a new line row to the DataFrame in-place."""
    new_idx = line_df.index.max() + 1 if len(line_df) > 0 else 0
    new_row = {
        'from_bus':   bus_i,
        'to_bus':     bus_j,
        'length_km':  length_km,
        'in_service': True,
        'name':       f"Cand_{bus_i}-{bus_j}",
    }
    new_row.update(line_params)
    line_df.loc[new_idx] = new_row


# ─────────────────────────────────────────────────────────────────────────────
# Main optimisation function
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_line_addition(
    grid:          WasterkingenGrid,
    bess_list:     list,
    pv_list:       list,
    load_list:     list,
    baseline,
    t_idx:         int   = 56,
    n_directions:  int   = 36,
    length_km:     float = 0.1,
    line_params:   dict  = None,
    top_n:         int   = 20,
    top_bus_count: int   = 20,
    verbose:       bool  = True,
):
    """
    Enumerate candidate bus pairs and return the one whose line addition
    maximises the convex-hull area of the single-timestep FFOR.
    """
    if verbose:
        print("\n[line_add] Computing baseline FFOR (no new line) ...")

    line_df_orig = grid.line_df.copy()

    grid.build_jacobian()
    base_opt  = FFOROptimizer(grid, bess_list, pv_list, load_list)
    base_ffor = base_opt.compute_ffor_single_timestep(
        t_idx=t_idx, baseline=baseline, n_directions=n_directions
    )
    base_area = _ffor_area(base_ffor.P_flex, base_ffor.Q_flex)
    if verbose:
        print(f"  Baseline FFOR area : {base_area:.6f} MW*MVAr")

    all_candidates = _candidate_pairs(grid.bus_df, line_df_orig, max_hops=3)
    if verbose:
        print(f"  Candidate pairs    : {len(all_candidates)} (<=3 hops, unconnected)")

    J_inv_base   = np.linalg.pinv(grid.J)
    non_pcc_list = [b for b in grid.bus_df.index if b != grid.pcc_bus]
    top_buses    = _top_significant_buses(J_inv_base, non_pcc_list, top_n=top_bus_count)
    filtered     = [(u, v) for u, v in all_candidates
                    if u in top_buses and v in top_buses]

    if filtered:
        all_candidates = filtered
        if verbose:
            print(f"  Restricting to {len(top_buses)} most significant buses "
                  f"-> {len(all_candidates)} pairs")
    else:
        if verbose:
            print("  No candidate pairs found among top buses; using all hop-filtered pairs")

    candidates = _score_candidates(all_candidates, J_inv_base, non_pcc_list, top_n=top_n)
    if verbose:
        print(f"  Evaluating top {len(candidates)} candidates (proxy-scored) ...")

    if line_params is None:
        line_params = _median_line_params(line_df_orig)

    results = []
    for idx, (bus_i, bus_j) in enumerate(candidates):
        if verbose:
            print(f"  [{idx+1}/{len(candidates)}] Testing edge ({bus_i}, {bus_j}) ...",
                  end="\r", flush=True)

        line_df_cand = line_df_orig.copy()
        _add_candidate_line(line_df_cand, bus_i, bus_j, length_km, line_params)

        grid.line_df = line_df_cand
        grid.n_lines = len(line_df_cand)
        grid.build_jacobian()

        try:
            opt  = FFOROptimizer(grid, bess_list, pv_list, load_list)
            ffor = opt.compute_ffor_single_timestep(
                t_idx=t_idx, baseline=baseline, n_directions=n_directions
            )
            area = _ffor_area(ffor.P_flex, ffor.Q_flex)
        except Exception as exc:
            if verbose:
                print(f"\n    [warn] edge ({bus_i},{bus_j}) failed: {exc}")
            area = 0.0
            ffor = None

        results.append({"bus_i": bus_i, "bus_j": bus_j, "area": area, "ffor": ffor})

    grid.line_df = line_df_orig
    grid.n_lines = len(line_df_orig)
    grid.build_jacobian()

    if verbose:
        print()

    results.sort(key=lambda r: r["area"], reverse=True)

    best      = results[0]
    best_area = best["area"]
    gain      = best_area - base_area
    gain_pct  = 100.0 * gain / base_area if base_area > 0 else float("inf")

    if verbose:
        print(f"\n  Best new line: bus {best['bus_i']} <-> bus {best['bus_j']}")
        print(f"  FFOR area  : {best_area:.6f} MW*MVAr")
        print(f"  Baseline   : {base_area:.6f} MW*MVAr")
        print(f"  Area gain  : {gain:+.6f} MW*MVAr ({gain_pct:+.1f}%)")

        print(f"\n  Top-5 candidates (by FFOR area):")
        print(f"  {'Rank':>4}  {'Bus i':>6}  {'Bus j':>6}  {'Area [MW*MVAr]':>16}  {'Gain [%]':>10}")
        for rank, r in enumerate(results[:5], 1):
            g = 100.0 * (r["area"] - base_area) / base_area if base_area > 0 else float("inf")
            print(f"  {rank:>4}  {r['bus_i']:>6}  {r['bus_j']:>6}  {r['area']:>16.6f}  {g:>+10.1f}")

    return {
        "best_pair":     (best["bus_i"], best["bus_j"]),
        "best_ffor":     best["ffor"],
        "best_area":     best_area,
        "base_area":     base_area,
        "area_gain":     gain,
        "area_gain_pct": gain_pct,
        "all_results":   results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helper
# ─────────────────────────────────────────────────────────────────────────────

def plot_ffor_comparison(base_ffor, opt_ffor, best_pair: tuple,
                         save_path: Path = None):
    """Overlay the baseline FFOR and the optimised FFOR."""
    from utils import _convex_hull_order, _close_polygon

    fig, ax = plt.subplots(figsize=(7, 7))

    for ffor, color, label in [
        (base_ffor, "steelblue", "Baseline FFOR"),
        (opt_ffor,  "tomato",    f"FFOR + line ({best_pair[0]}<->{best_pair[1]})"),
    ]:
        if ffor is None or len(ffor.P_flex) < 2:
            continue
        P, Q = _convex_hull_order(ffor.P_flex, ffor.Q_flex)
        P, Q = _close_polygon(P, Q)
        ax.fill(P, Q, alpha=0.20, color=color)
        ax.plot(P, Q, "-", color=color, linewidth=1.8, label=label)

    ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.6, linestyle="--")
    ax.plot(0, 0, "ko", markersize=6, label="Baseline operating point")
    ax.set_xlabel("Active Power Flexibility [MW]")
    ax.set_ylabel("Reactive Power Flexibility [MVAr]")
    ax.set_title("FFOR: baseline vs. optimal line addition")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)
