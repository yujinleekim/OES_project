"""
line_add.py
───────────
Find the optimal location between two unconnected vertices in WasterkingenGrid
to add an edge (line) such that the FFOR area is maximised.

Approach
────────
1. Pre-filter candidates to the 20 most electrically significant pairs using
   a fast proxy score (sum of |J_inv| sensitivity rows for both buses).
2. For each of the 20 candidates, temporarily add the new line, rebuild the
   Jacobian (Y-matrix only — no runpp needed), and recompute the FFOR.
3. Score each candidate by the convex-hull area of the resulting FFOR polygon.
4. Report the winning pair and the area gain vs baseline.
"""

import copy
import itertools
import numpy as np
import pandapower as pp
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
    mask = ~(np.isnan(P) | np.isnan(Q))
    pts  = np.column_stack([P[mask], Q[mask]])
    if len(pts) < 3:
        return 0.0
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Candidate-line helpers
# ─────────────────────────────────────────────────────────────────────────────

def _existing_edges(net: pp.pandapowerNet) -> set:
    edges = set()
    for _, row in net.line.iterrows():
        u, v = int(row.from_bus), int(row.to_bus)
        edges.add((min(u, v), max(u, v)))
    return edges


def _hop_candidates(net: pp.pandapowerNet, max_hops: int = 3) -> list:
    """
    Unconnected pairs within max_hops of each other (networkx graph walk).
    Falls back to all pairs if network is unavailable.
    """
    try:
        import networkx as nx
    except ImportError:
        bus_indices = list(net.bus.index)
        existing    = _existing_edges(net)
        return [(u, v)
                for u, v in itertools.combinations(bus_indices, 2)
                if (min(u, v), max(u, v)) not in existing]

    G = nx.Graph()
    for _, row in net.line.iterrows():
        G.add_edge(int(row.from_bus), int(row.to_bus))
    existing = _existing_edges(net)
    candidates = []
    for u in list(G.nodes()):
        lengths = nx.single_source_shortest_path_length(G, u, cutoff=max_hops)
        for v, hops in lengths.items():
            if v <= u or hops < 2:
                continue
            if (min(u, v), max(u, v)) not in existing:
                candidates.append((u, v))
    return candidates


def _score_candidates(candidates: list, J_inv: np.ndarray,
                      non_pcc_positions: list, top_n: int = 20) -> list:
    """
    Fast proxy score: for each candidate pair (i, j), estimate how much the
    new line will change the FFOR by looking at the L1 norm of the voltage-
    sensitivity rows for both buses.  Buses with large sensitivity to injections
    are the most constrained and benefit most from a new interconnection.

    Returns the top_n candidates sorted by score descending.
    """
    n_pq   = len(non_pcc_positions)
    J_volt = J_inv[n_pq:]        # voltage sensitivity block  (n_pq × 2*n_pq)

    # Map bus index → position in non_pcc list
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
    """
    Rank buses by the L1 norm of their voltage-sensitivity rows in J_inv.
    The highest-scoring buses are the most electrically significant
    candidates for a new interconnection.
    """
    n_pq   = len(non_pcc_positions)
    J_volt = J_inv[n_pq:]
    pos_map = {bus: p for p, bus in enumerate(non_pcc_positions)}

    scored = []
    for bus in non_pcc_positions:
        p = pos_map[bus]
        scored.append((float(np.sum(np.abs(J_volt[p]))), bus))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [bus for _, bus in scored[:top_n]]


def _median_line_params(net: pp.pandapowerNet) -> dict:
    lines = net.line
    return {
        "r_ohm_per_km": float(np.median(lines.r_ohm_per_km)),
        "x_ohm_per_km": float(np.median(lines.x_ohm_per_km)),
        "c_nf_per_km":  float(np.median(lines.c_nf_per_km)),
        "max_i_ka":     float(np.median(lines.max_i_ka)),
    }


def _add_candidate_line(net, bus_i, bus_j, length_km, line_params):
    return pp.create_line_from_parameters(
        net, from_bus=bus_i, to_bus=bus_j,
        length_km=length_km, name=f"Cand_{bus_i}-{bus_j}",
        **line_params,
    )


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
    Pre-filter to the top_n most electrically significant candidate pairs,
    then evaluate each with a full FFOR solve and return the best.
    """
    net_orig = grid.net

    # ── 0. baseline FFOR ──────────────────────────────────────────────────────
    if verbose:
        print("\n[line_add] Computing baseline FFOR (no new line) ...")

    # Ensure Jacobian is built for the original network.
    # Build directly (no runpp) so we never touch the raw ndarray columns.
    grid.J, grid.J_blocks = _build_jacobian(net_orig)

    base_opt  = FFOROptimizer(grid, bess_list, pv_list, load_list)
    base_ffor = base_opt.compute_ffor_single_timestep(
        t_idx=t_idx, baseline=baseline, n_directions=n_directions
    )
    base_area = _ffor_area(base_ffor.P_flex, base_ffor.Q_flex)
    if verbose:
        print(f"  Baseline FFOR area : {base_area:.6f} MW·MVAr")

    # ── 1. hop-filtered candidates → top-vertex prefilter → proxy-scored top_n ──
    all_candidates = _hop_candidates(net_orig, max_hops=3)
    if verbose:
        print(f"  Hop-filtered pairs : {len(all_candidates)} (≤3 hops, unconnected)")

    J_inv_base    = np.linalg.pinv(grid.J)
    non_pcc_list  = [b for b in net_orig.bus.index if b != grid.pcc_bus]
    top_buses     = _top_significant_buses(J_inv_base, non_pcc_list, top_n=top_bus_count)
    filtered = [(u, v) for u, v in all_candidates if u in top_buses and v in top_buses]
    if filtered:
        all_candidates = filtered
        if verbose:
            print(f"  Restricting to {len(top_buses)} most significant buses -> {len(all_candidates)} pairs")
    else:
        if verbose:
            print("  No candidate pairs found among top buses; using all hop-filtered pairs")

    candidates = _score_candidates(all_candidates, J_inv_base,
                                   non_pcc_list, top_n=top_n)
    if verbose:
        print(f"  Evaluating top {len(candidates)} candidates (proxy-scored) ...")

    if line_params is None:
        line_params = _median_line_params(net_orig)

    # ── 2. evaluate each top candidate ────────────────────────────────────────
    results = []
    for idx, (bus_i, bus_j) in enumerate(candidates):
        if verbose:
            print(f"  [{idx+1}/{len(candidates)}] Testing edge ({bus_i}, {bus_j}) ...",
                  end="\r", flush=True)

        # Deep-copy network and immediately repair DataFrame columns so that
        # any subsequent pp.runpp calls (if ever attempted) won't crash on the
        # raw numpy array columns preserved by deepcopy.
        net_copy = copy.deepcopy(net_orig)
        WasterkingenGrid._repair_dataframes(net_copy)

        _add_candidate_line(net_copy, bus_i, bus_j, length_km, line_params)

        # Swap into grid object and rebuild Jacobian directly from Y-matrix
        # (no runpp — avoids the fillna crash entirely in the candidate loop)
        grid.net     = net_copy
        grid.n_lines = len(net_copy.line)
        grid.J, grid.J_blocks = _build_jacobian(net_copy)

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

    # ── 3. restore original network ───────────────────────────────────────────
    grid.net     = net_orig
    grid.n_lines = len(net_orig.line)
    grid.J, grid.J_blocks = _build_jacobian(net_orig)

    if verbose:
        print()

    # ── 4. rank and report ────────────────────────────────────────────────────
    results.sort(key=lambda r: r["area"], reverse=True)
    best      = results[0]
    best_area = best["area"]
    gain      = best_area - base_area
    gain_pct  = 100.0 * gain / base_area if base_area > 0 else float("inf")

    if verbose:
        print(f"\n  ╔══════════════════════════════════════════╗")
        print(f"  ║  OPTIMAL NEW LINE FOUND                  ║")
        print(f"  ╠══════════════════════════════════════════╣")
        print(f"  ║  Bus pair   : {best['bus_i']:>4d}  ↔  {best['bus_j']:<4d}            ║")
        print(f"  ║  FFOR area  : {best_area:>10.6f} MW·MVAr         ║")
        print(f"  ║  Baseline   : {base_area:>10.6f} MW·MVAr         ║")
        print(f"  ║  Area gain  : {gain:>+10.6f} MW·MVAr ({gain_pct:+.1f}%) ║")
        print(f"  ╚══════════════════════════════════════════╝")
        print(f"\n  Top-5 candidates:")
        print(f"  {'Rank':>4}  {'Bus i':>6}  {'Bus j':>6}  {'Area [MW·MVAr]':>16}  {'Gain [%]':>10}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*16}  {'─'*10}")
        for rank, r in enumerate(results[:5], 1):
            g = 100.0*(r["area"]-base_area)/base_area if base_area > 0 else float("inf")
            print(f"  {rank:>4}  {r['bus_i']:>6}  {r['bus_j']:>6}"
                  f"  {r['area']:>16.6f}  {g:>+10.1f}")

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
    from utils import _convex_hull_order, _close_polygon
    fig, ax = plt.subplots(figsize=(7, 7))
    for ffor, color, label in [
        (base_ffor, "steelblue", "Baseline FFOR"),
        (opt_ffor,  "tomato",    f"FFOR + line ({best_pair[0]}↔{best_pair[1]})"),
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
"""
Approach
────────
1. Enumerate all pairs of buses that do NOT share an existing line.
2. For each candidate pair (i, j), temporarily add a new line to the
   pandapower network, rebuild the Jacobian, and recompute the FFOR at
   a chosen operating point (single-timestep).
3. Score each candidate by the convex-hull area of the resulting FFOR polygon.
4. Report the winning pair and the area gain compared to the baseline FFOR.

Line parameters for the candidate edge
────────────────────────────────────────
The new cable is assumed to be of the same type as the median existing line
(r, x, c per km) and a default length of 0.1 km (100 m, typical LV feeder
branch).  Both the length and impedance can be overridden via keyword
arguments to find_optimal_line_addition().
"""

import copy
import itertools
import numpy as np
import pandapower as pp
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import ConvexHull

from grid_model import WasterkingenGrid
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

def _existing_edges(net: pp.pandapowerNet) -> set:
    """Return the set of unordered bus-index pairs that already have a line."""
    edges = set()
    for _, row in net.line.iterrows():
        u, v = int(row.from_bus), int(row.to_bus)
        edges.add((min(u, v), max(u, v)))
    return edges


def _candidate_pairs(net: pp.pandapowerNet, max_hops: int = 3) -> list:
    """
    Return candidate (bus_i, bus_j) pairs for new line addition.

    Only considers pairs that are:
      - Not already connected by a line
      - Within max_hops of each other in the existing graph
        (adding a long-distance tie in a radial LV grid has negligible
        effect on FFOR and greatly inflates the search space)
      - At least 2 hops apart (1-hop pairs already have a direct line)

    For a 195-bus radial network this reduces candidates from ~18,700
    to typically 200–600, making the search feasible in minutes.
    """
    try:
        import networkx as nx
    except ImportError:
        # networkx not available — fall back to all pairs (slow but correct)
        bus_indices = list(net.bus.index)
        existing    = _existing_edges(net)
        return [(u, v)
                for u, v in itertools.combinations(bus_indices, 2)
                if (min(u, v), max(u, v)) not in existing]

    G = nx.Graph()
    for _, row in net.line.iterrows():
        G.add_edge(int(row.from_bus), int(row.to_bus))

    existing = _existing_edges(net)
    candidates = []
    for u in list(G.nodes()):
        lengths = nx.single_source_shortest_path_length(G, u, cutoff=max_hops)
        for v, hops in lengths.items():
            if v <= u:
                continue
            if hops < 2:
                continue   # already directly connected
            if (min(u, v), max(u, v)) in existing:
                continue
            candidates.append((u, v))
    return candidates


def _median_line_params(net: pp.pandapowerNet) -> dict:
    """
    Extract representative per-km line parameters from the existing network
    (median over all lines, so a single outlier cable doesn't skew results).
    """
    lines = net.line
    return {
        "r_ohm_per_km": float(np.median(lines.r_ohm_per_km)),
        "x_ohm_per_km": float(np.median(lines.x_ohm_per_km)),
        "c_nf_per_km":  float(np.median(lines.c_nf_per_km)),
        "max_i_ka":     float(np.median(lines.max_i_ka)),
    }


def _add_candidate_line(net: pp.pandapowerNet,
                         bus_i: int, bus_j: int,
                         length_km: float,
                         line_params: dict) -> int:
    """
    Temporarily add a line between bus_i and bus_j.
    Returns the new line index.
    """
    return pp.create_line_from_parameters(
        net,
        from_bus=bus_i,
        to_bus=bus_j,
        length_km=length_km,
        name=f"Cand_{bus_i}-{bus_j}",
        **line_params,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main optimisation function
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_line_addition(
    grid: WasterkingenGrid,
    bess_list: list,
    pv_list:   list,
    load_list: list,
    baseline,
    t_idx:        int   = 56,        # timestep index for FFOR evaluation (14:00)
    n_directions: int   = 36,        # angular resolution for FFOR scan
    length_km:    float = 0.1,       # assumed length of the new cable [km]
    line_params:  dict  = None,      # override cable impedance if desired
    verbose:      bool  = True,
    top_n:         int   = 20,      # how many proxy-scored candidates to evaluate
    top_bus_count: int   = 20,      # prefilter to the N most significant buses
):
    """
    Enumerate all unconnected bus pairs and return the one whose addition
    maximises the convex-hull area of the single-timestep FFOR.

    Parameters
    ----------
    grid        : WasterkingenGrid instance (will NOT be mutated).
    bess_list   : list of BESSModel objects.
    pv_list     : list of PVModel objects.
    load_list   : list of ControllableLoadModel objects.
    baseline    : DataFrame from grid.get_baseline().
    t_idx       : timestep at which the FFOR is evaluated.
    n_directions: number of scanning directions for the FFOR LP.
    length_km   : assumed length of the candidate new line.
    line_params : dict of per-km parameters for the new line; uses median of
                  existing lines if None.
    verbose     : print progress to stdout.

    Returns
    -------
    dict with keys:
        best_pair   – (bus_i, bus_j) pandapower bus indices
        best_area   – FFOR area with the new line [MW·MVAr]
        base_area   – FFOR area without any new line [MW·MVAr]
        area_gain   – absolute gain in area [MW·MVAr]
        area_gain_pct – percentage gain [%]
        all_results – list of dicts for every candidate, sorted by area desc.
    """
    net_orig = grid.net

    # ── 0. baseline FFOR (no new line) ────────────────────────────────────────
    if verbose:
        print("\n[line_add] Computing baseline FFOR (no new line) ...")
    grid.build_jacobian()
    base_opt  = FFOROptimizer(grid, bess_list, pv_list, load_list)
    base_ffor = base_opt.compute_ffor_single_timestep(
        t_idx=t_idx, baseline=baseline, n_directions=n_directions
    )
    base_area = _ffor_area(base_ffor.P_flex, base_ffor.Q_flex)
    if verbose:
        print(f"  Baseline FFOR area : {base_area:.6f} MW·MVAr")

    # ── 1. hop-filtered candidates → top-vertex prefilter → proxy-scored top_n ──
    all_candidates = _candidate_pairs(net_orig, max_hops=3)
    if verbose:
        print(f"  Candidate pairs    : {len(all_candidates)} (≤3 hops, unconnected)")

    # Build Jacobian and compute a fast proxy score to prefilter
    J_inv_base   = np.linalg.pinv(grid.J)
    non_pcc_list = [b for b in net_orig.bus.index if b != grid.pcc_bus]
    top_buses    = _top_significant_buses(J_inv_base, non_pcc_list, top_n=top_bus_count)
    filtered = [(u, v) for u, v in all_candidates if u in top_buses and v in top_buses]
    if filtered:
        all_candidates = filtered
        if verbose:
            print(f"  Restricting to {len(top_buses)} most significant buses -> {len(all_candidates)} pairs")
    else:
        if verbose:
            print("  No candidate pairs found among top buses; using all hop-filtered pairs")

    candidates = _score_candidates(all_candidates, J_inv_base,
                                   non_pcc_list, top_n=top_n)
    if verbose:
        print(f"  Evaluating top {len(candidates)} candidates (proxy-scored) ...")

    if line_params is None:
        line_params = _median_line_params(net_orig)

    # ── 2. evaluate each candidate ────────────────────────────────────────────
    results = []
    for idx, (bus_i, bus_j) in enumerate(candidates):
        if verbose:
            print(f"  [{idx+1}/{len(candidates)}] Testing edge ({bus_i}, {bus_j}) ...",
                  end="\r", flush=True)

        # deep-copy the pandapower network so the original is never touched
        net_copy = copy.deepcopy(net_orig)

        # repair DataFrame columns on the copy — deep-copy preserves raw numpy
        # arrays from the Excel load, which cause runpp's .fillna() to crash
        WasterkingenGrid._repair_dataframes(net_copy)

        # add the candidate line
        _add_candidate_line(net_copy, bus_i, bus_j, length_km, line_params)

        # temporarily swap the network into the grid object
        grid.net     = net_copy
        grid.n_lines = len(net_copy.line)

        # rebuild Jacobian for the modified topology
        grid.build_jacobian()

        # re-instantiate the optimizer (it pre-computes J_inv and H_branch)
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

        results.append({
            "bus_i":  bus_i,
            "bus_j":  bus_j,
            "area":   area,
            "ffor":   ffor,
        })

    # restore original network
    grid.net     = net_orig
    grid.n_lines = len(net_orig.line)
    grid.build_jacobian()

    if verbose:
        print()   # newline after \r progress

    # ── 3. rank and report ────────────────────────────────────────────────────
    results.sort(key=lambda r: r["area"], reverse=True)

    best       = results[0]
    best_area  = best["area"]
    gain       = best_area - base_area
    gain_pct   = 100.0 * gain / base_area if base_area > 0 else float("inf")

    if verbose:
        print(f"\n  ╔══════════════════════════════════════════╗")
        print(f"  ║  OPTIMAL NEW LINE FOUND                  ║")
        print(f"  ╠══════════════════════════════════════════╣")
        print(f"  ║  Bus pair   : {best['bus_i']:>4d}  ↔  {best['bus_j']:<4d}            ║")
        print(f"  ║  FFOR area  : {best_area:>10.6f} MW·MVAr         ║")
        print(f"  ║  Baseline   : {base_area:>10.6f} MW·MVAr         ║")
        print(f"  ║  Area gain  : {gain:>+10.6f} MW·MVAr ({gain_pct:+.1f}%) ║")
        print(f"  ╚══════════════════════════════════════════╝")

        # top-5 candidates
        print(f"\n  Top-5 candidates (by FFOR area):")
        print(f"  {'Rank':>4}  {'Bus i':>6}  {'Bus j':>6}  {'Area [MW·MVAr]':>16}  {'Gain [%]':>10}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*16}  {'─'*10}")
        for rank, r in enumerate(results[:5], 1):
            g = 100.0 * (r["area"] - base_area) / base_area if base_area > 0 else float("inf")
            print(f"  {rank:>4}  {r['bus_i']:>6}  {r['bus_j']:>6}  {r['area']:>16.6f}  {g:>+10.1f}")

    return {
        "best_pair":      (best["bus_i"], best["bus_j"]),
        "best_ffor":      best["ffor"],
        "best_area":      best_area,
        "base_area":      base_area,
        "area_gain":      gain,
        "area_gain_pct":  gain_pct,
        "all_results":    results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helper
# ─────────────────────────────────────────────────────────────────────────────

def plot_ffor_comparison(base_ffor, opt_ffor, best_pair: tuple,
                         save_path: Path = None):
    """
    Overlay the baseline FFOR and the optimised FFOR (after adding the best
    new line) on the same axes.
    """
    from utils import _convex_hull_order, _close_polygon

    fig, ax = plt.subplots(figsize=(7, 7))

    for ffor, color, label in [
        (base_ffor, "steelblue", "Baseline FFOR"),
        (opt_ffor,  "tomato",    f"FFOR + line ({best_pair[0]}↔{best_pair[1]})"),
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


def _median_line_params(net: pp.pandapowerNet) -> dict:
    """
    Extract representative per-km line parameters from the existing network
    (median over all lines, so a single outlier cable doesn't skew results).
    """
    lines = net.line
    return {
        "r_ohm_per_km": float(np.median(lines.r_ohm_per_km)),
        "x_ohm_per_km": float(np.median(lines.x_ohm_per_km)),
        "c_nf_per_km":  float(np.median(lines.c_nf_per_km)),
        "max_i_ka":     float(np.median(lines.max_i_ka)),
    }


def _add_candidate_line(net: pp.pandapowerNet,
                         bus_i: int, bus_j: int,
                         length_km: float,
                         line_params: dict) -> int:
    """
    Temporarily add a line between bus_i and bus_j.
    Returns the new line index.
    """
    return pp.create_line_from_parameters(
        net,
        from_bus=bus_i,
        to_bus=bus_j,
        length_km=length_km,
        name=f"Cand_{bus_i}-{bus_j}",
        **line_params,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main optimisation function
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_line_addition(
    grid: WasterkingenGrid,
    bess_list: list,
    pv_list:   list,
    load_list: list,
    baseline,
    t_idx:        int   = 56,        # timestep index for FFOR evaluation (14:00)
    n_directions: int   = 36,        # angular resolution for FFOR scan
    length_km:    float = 0.1,       # assumed length of the new cable [km]
    line_params:  dict  = None,      # override cable impedance if desired
    verbose:      bool  = True,
    top_n:         int   = 20,      # how many proxy-scored candidates to evaluate
    top_bus_count: int   = 20,      # prefilter to the N most significant buses
):
    """
    Enumerate all unconnected bus pairs and return the one whose addition
    maximises the convex-hull area of the single-timestep FFOR.

    Parameters
    ----------
    grid        : WasterkingenGrid instance (will NOT be mutated).
    bess_list   : list of BESSModel objects.
    pv_list     : list of PVModel objects.
    load_list   : list of ControllableLoadModel objects.
    baseline    : DataFrame from grid.get_baseline().
    t_idx       : timestep at which the FFOR is evaluated.
    n_directions: number of scanning directions for the FFOR LP.
    length_km   : assumed length of the candidate new line.
    line_params : dict of per-km parameters for the new line; uses median of
                  existing lines if None.
    verbose     : print progress to stdout.

    Returns
    -------
    dict with keys:
        best_pair   – (bus_i, bus_j) pandapower bus indices
        best_area   – FFOR area with the new line [MW·MVAr]
        base_area   – FFOR area without any new line [MW·MVAr]
        area_gain   – absolute gain in area [MW·MVAr]
        area_gain_pct – percentage gain [%]
        all_results – list of dicts for every candidate, sorted by area desc.
    """
    net_orig = grid.net

    # ── 0. baseline FFOR (no new line) ────────────────────────────────────────
    if verbose:
        print("\n[line_add] Computing baseline FFOR (no new line) ...")
    grid.build_jacobian()
    base_opt  = FFOROptimizer(grid, bess_list, pv_list, load_list)
    base_ffor = base_opt.compute_ffor_single_timestep(
        t_idx=t_idx, baseline=baseline, n_directions=n_directions
    )
    base_area = _ffor_area(base_ffor.P_flex, base_ffor.Q_flex)
    if verbose:
        print(f"  Baseline FFOR area : {base_area:.6f} MW·MVAr")

    # ── 1. hop-filtered candidates → top-vertex prefilter → proxy-scored top_n ──
    all_candidates = _candidate_pairs(net_orig, max_hops=3)
    if verbose:
        print(f"  Candidate pairs    : {len(all_candidates)} (≤3 hops, unconnected)")

    # Build Jacobian and compute a fast proxy score to prefilter
    J_inv_base   = np.linalg.pinv(grid.J)
    non_pcc_list = [b for b in net_orig.bus.index if b != grid.pcc_bus]
    top_buses    = _top_significant_buses(J_inv_base, non_pcc_list, top_n=20)
    filtered = [(u, v) for u, v in all_candidates if u in top_buses and v in top_buses]
    if filtered:
        all_candidates = filtered
        if verbose:
            print(f"  Restricting to {len(top_buses)} most significant buses -> {len(all_candidates)} pairs")
    else:
        if verbose:
            print("  No candidate pairs found among top buses; using all hop-filtered pairs")

    candidates = _score_candidates(all_candidates, J_inv_base,
                                   non_pcc_list, top_n=top_n)
    if verbose:
        print(f"  Evaluating top {len(candidates)} candidates (proxy-scored) ...")

    if line_params is None:
        line_params = _median_line_params(net_orig)

    # ── 2. evaluate each candidate ────────────────────────────────────────────
    results = []
    for idx, (bus_i, bus_j) in enumerate(candidates):
        if verbose:
            print(f"  [{idx+1}/{len(candidates)}] Testing edge ({bus_i}, {bus_j}) ...",
                  end="\r", flush=True)

        # deep-copy the pandapower network so the original is never touched
        net_copy = copy.deepcopy(net_orig)

        # repair DataFrame columns on the copy — deep-copy preserves raw numpy
        # arrays from the Excel load, which cause runpp's .fillna() to crash
        WasterkingenGrid._repair_dataframes(net_copy)

        # add the candidate line
        _add_candidate_line(net_copy, bus_i, bus_j, length_km, line_params)

        # temporarily swap the network into the grid object
        grid.net     = net_copy
        grid.n_lines = len(net_copy.line)

        # rebuild Jacobian for the modified topology
        grid.build_jacobian()

        # re-instantiate the optimizer (it pre-computes J_inv and H_branch)
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

        results.append({
            "bus_i":  bus_i,
            "bus_j":  bus_j,
            "area":   area,
            "ffor":   ffor,
        })

    # restore original network
    grid.net     = net_orig
    grid.n_lines = len(net_orig.line)
    grid.build_jacobian()

    if verbose:
        print()   # newline after \r progress

    # ── 3. rank and report ────────────────────────────────────────────────────
    results.sort(key=lambda r: r["area"], reverse=True)

    best       = results[0]
    best_area  = best["area"]
    gain       = best_area - base_area
    gain_pct   = 100.0 * gain / base_area if base_area > 0 else float("inf")

    if verbose:
        print(f"\n  ╔══════════════════════════════════════════╗")
        print(f"  ║  OPTIMAL NEW LINE FOUND                  ║")
        print(f"  ╠══════════════════════════════════════════╣")
        print(f"  ║  Bus pair   : {best['bus_i']:>4d}  ↔  {best['bus_j']:<4d}            ║")
        print(f"  ║  FFOR area  : {best_area:>10.6f} MW·MVAr         ║")
        print(f"  ║  Baseline   : {base_area:>10.6f} MW·MVAr         ║")
        print(f"  ║  Area gain  : {gain:>+10.6f} MW·MVAr ({gain_pct:+.1f}%) ║")
        print(f"  ╚══════════════════════════════════════════╝")

        # top-5 candidates
        print(f"\n  Top-5 candidates (by FFOR area):")
        print(f"  {'Rank':>4}  {'Bus i':>6}  {'Bus j':>6}  {'Area [MW·MVAr]':>16}  {'Gain [%]':>10}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*16}  {'─'*10}")
        for rank, r in enumerate(results[:5], 1):
            g = 100.0 * (r["area"] - base_area) / base_area if base_area > 0 else float("inf")
            print(f"  {rank:>4}  {r['bus_i']:>6}  {r['bus_j']:>6}  {r['area']:>16.6f}  {g:>+10.1f}")

    return {
        "best_pair":      (best["bus_i"], best["bus_j"]),
        "best_ffor":      best["ffor"],
        "best_area":      best_area,
        "base_area":      base_area,
        "area_gain":      gain,
        "area_gain_pct":  gain_pct,
        "all_results":    results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helper
# ─────────────────────────────────────────────────────────────────────────────

def plot_ffor_comparison(base_ffor, opt_ffor, best_pair: tuple,
                         save_path: Path = None):
    """
    Overlay the baseline FFOR and the optimised FFOR (after adding the best
    new line) on the same axes.
    """
    from utils import _convex_hull_order, _close_polygon

    fig, ax = plt.subplots(figsize=(7, 7))

    for ffor, color, label in [
        (base_ffor, "steelblue", "Baseline FFOR"),
        (opt_ffor,  "tomato",    f"FFOR + line ({best_pair[0]}↔{best_pair[1]})"),
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