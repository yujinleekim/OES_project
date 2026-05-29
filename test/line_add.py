"""
line_add.py
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

import gurobipy as gp
from gurobipy import GRB

class ShadowPriceScorer(FFOROptimizer):
    def rank_candidates_by_shadow_price(self, candidates, t_idx, baseline, n_directions=36):
        angles_dir = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
        pos_map = {bus: pos for pos, bus in enumerate(self._non_pcc)}
        pair_scores = {pair: 0.0 for pair in candidates}
        valid_angles = 0
        
        n_pq, n_lines, n_sides = self._n_pq, len(self.grid.line_df), 8
        poly_angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        delta = np.pi / n_sides
        
        for a in angles_dir:
            cos_th, sin_th = np.cos(a), np.sin(a)
            m = gp.Model()
            m.setParam("OutputFlag", 0)
            
            dt_h, irr = 0.25, self._irr(baseline, t_idx)

            bess_P, bess_Q = [], []
            for i, bess in enumerate(self.bess_list):
                p = m.addVar(lb=-bess.S_max, ub=bess.S_max, name=f"b{i}P")
                q = m.addVar(lb=-bess.S_max, ub=bess.S_max, name=f"b{i}Q")
                bess.add_single_timestep_constraints(m, p, q, dt_h=dt_h, name=f"b{i}")
                bess_P.append(p); bess_Q.append(q)

            pv_P, pv_Q = [], []
            for i, pv in enumerate(self.pv_list):
                p = m.addVar(lb=0, ub=pv.p_max(irr), name=f"pv{i}P")
                q = m.addVar(lb=-pv.S_max * pv.tan_phi, ub=pv.S_max * pv.tan_phi, name=f"pv{i}Q")
                pv.add_constraints(m, p, q, irradiance=irr, name=f"pv{i}")
                pv_P.append(p); pv_Q.append(q)

            load_P, load_Q = [], []
            for i, ld in enumerate(self.load_list):
                p = m.addVar(lb=0, ub=ld.P_max, name=f"l{i}P")
                q = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"l{i}Q")
                ld.add_single_timestep_constraints(m, p, q, P_base=0.0, name=f"l{i}")
                load_P.append(p); load_Q.append(q)

            m.update()
            dP, dQ = self._injection_deviations(bess_P, bess_Q, pv_P, pv_Q, load_P, load_Q, baseline, t_idx)
            dP_pcc = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="dP_pcc")
            dQ_pcc = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="dQ_pcc")
            self._add_pcc_balance(m, dP, dQ, dP_pcc, dQ_pcc)
            
            self._add_voltage_constraints(m, dP, dQ)
            self._add_line_constraints(m, dP, dQ)
            
            m.update()
            m.setObjective(cos_th * dP_pcc + sin_th * dQ_pcc, GRB.MAXIMIZE)
            m.optimize()
            
            if m.Status == GRB.OPTIMAL:
                valid_angles += 1
                lmp_P = {pos: 0.0 for pos in range(n_pq)}
                for b in range(n_pq):
                    try:
                        pi_lo, pi_hi = m.getConstrByName(f"Vlo{b}").Pi, m.getConstrByName(f"Vhi{b}").Pi
                    except AttributeError:
                        pi_lo, pi_hi = 0.0, 0.0
                    if pi_lo != 0 or pi_hi != 0:
                        for pos in range(n_pq):
                            sens = float(self._J_inv[n_pq + b, pos])
                            lmp_P[pos] += abs(pi_lo * sens) + abs(pi_hi * sens)
                            
                for l in range(n_lines):
                    for k in range(n_sides):
                        try:
                            pi_ln = m.getConstrByName(f"Ln{l}e{k}").Pi
                        except AttributeError:
                            pi_ln = 0.0
                        if pi_ln != 0:
                            a_mid = poly_angles[k] + delta
                            nx, ny = np.cos(a_mid), np.sin(a_mid)
                            for pos in range(n_pq):
                                sens = nx * float(self._H_branch[l, pos]) + ny * float(self._H_branch[n_lines + l, pos])
                                lmp_P[pos] += abs(pi_ln * sens)
                                
                for (u, v) in candidates:
                    pu, pv = pos_map.get(u, -1), pos_map.get(v, -1)
                    lmp_u = lmp_P[pu] if pu >= 0 else 0.0
                    lmp_v = lmp_P[pv] if pv >= 0 else 0.0
                    pair_scores[(u, v)] += abs(lmp_u - lmp_v)
                    
        if valid_angles > 0:
            for pair in pair_scores: pair_scores[pair] /= valid_angles
                
        ranked = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
        return [pair for pair, score in ranked]

def _ffor_area(P: np.ndarray, Q: np.ndarray) -> float:
    mask = ~(np.isnan(P) | np.isnan(Q))
    pts  = np.column_stack([P[mask], Q[mask]])
    pts = np.unique(np.round(pts, 6), axis=0)
    if len(pts) < 3: return 0.0
    try:
        return ConvexHull(pts, qhull_options='QJ').volume
    except Exception: 
        return 0.0

def _existing_edges(line_df: pd.DataFrame) -> set:
    edges = set()
    for _, row in line_df.iterrows():
        u, v = int(row['from_bus']), int(row['to_bus'])
        edges.add((min(u, v), max(u, v)))
    return edges

def _candidate_pairs(bus_df: pd.DataFrame, line_df: pd.DataFrame, min_hops: int = 6) -> list:
    import networkx as nx
    G = nx.Graph()
    for _, row in line_df.iterrows():
        if row.get('in_service', True): G.add_edge(int(row['from_bus']), int(row['to_bus']))
    existing = _existing_edges(line_df)
    candidates = []
    lengths = dict(nx.all_pairs_shortest_path_length(G))

    for u in G.nodes():
        for v in G.nodes():
            if v <= u: continue
            hops = lengths.get(u, {}).get(v, float('inf'))
            if hops < min_hops: continue
            if (min(u, v), max(u, v)) in existing: continue
            candidates.append((u, v))
    return candidates

def _median_line_params(line_df: pd.DataFrame) -> dict:
    return {
        "r_ohm_per_km": float(np.median(line_df['r_ohm_per_km'])),
        "x_ohm_per_km": float(np.median(line_df['x_ohm_per_km'])),
        "c_nf_per_km":  float(np.median(line_df['c_nf_per_km'])),
        "max_i_ka":     float(np.median(line_df['max_i_ka'])),
    }

def _add_candidate_line(line_df: pd.DataFrame, bus_i: int, bus_j: int, length_km: float, line_params: dict):
    new_idx = line_df.index.max() + 1 if len(line_df) > 0 else 0
    new_row = {
        'from_bus':   bus_i, 'to_bus':     bus_j,
        'length_km':  length_km, 'in_service': True,
        'name':       f"Cand_{bus_i}-{bus_j}",
    }
    new_row.update(line_params)
    line_df.loc[new_idx] = new_row

def find_optimal_line_addition(
    grid: WasterkingenGrid, bess_list: list, pv_list: list, load_list: list, baseline,
    t_idx: int = 56, n_directions: int = 36, length_km: float = 0.1, line_params: dict = None,
    top_n: int = 20, verbose: bool = True
):
    line_df_orig = grid.line_df.copy()
    grid.build_jacobian()
    base_opt  = FFOROptimizer(grid, bess_list, pv_list, load_list)
    base_ffor = base_opt.compute_ffor_single_timestep(t_idx=t_idx, baseline=baseline, n_directions=n_directions)
    base_area = _ffor_area(base_ffor.P_flex, base_ffor.Q_flex)

    all_candidates = _candidate_pairs(grid.bus_df, line_df_orig, min_hops=8)
    scorer = ShadowPriceScorer(grid, bess_list, pv_list, load_list)
    ranked_candidates = scorer.rank_candidates_by_shadow_price(all_candidates, t_idx, baseline, n_directions)
    candidates = ranked_candidates[:top_n]

    if line_params is None: line_params = _median_line_params(line_df_orig)
    results = []
    
    for idx, (bus_i, bus_j) in enumerate(candidates):
        line_df_cand = line_df_orig.copy()
        _add_candidate_line(line_df_cand, bus_i, bus_j, length_km, line_params)
        grid.line_df, grid.n_lines = line_df_cand, len(line_df_cand)
        grid.build_jacobian()

        try:
            opt  = FFOROptimizer(grid, bess_list, pv_list, load_list)
            ffor = opt.compute_ffor_single_timestep(t_idx=t_idx, baseline=baseline, n_directions=n_directions)
            area = _ffor_area(ffor.P_flex, ffor.Q_flex)
        except Exception:
            area, ffor = 0.0, None

        results.append({"bus_i": bus_i, "bus_j": bus_j, "area": area, "ffor": ffor})

    grid.line_df, grid.n_lines = line_df_orig, len(line_df_orig)
    grid.build_jacobian()

    if not results: return {}
    results.sort(key=lambda r: r["area"], reverse=True)
    best      = results[0]
    best_area = best["area"]
    gain      = best_area - base_area
    gain_pct  = 100.0 * gain / base_area if base_area > 0 else float("inf")

    return {
        "best_pair": (best["bus_i"], best["bus_j"]), "best_ffor": best["ffor"],
        "best_area": best_area, "base_area": base_area,
        "area_gain": gain, "area_gain_pct": gain_pct,
        "all_results": results,
    }

# In OES_project-3/test/line_add.py
def plot_ffor_comparison(base_ffor, opt_ffor, best_pair, save_path=None):
    # ... (dein Plotting Code) ...
    from utils import _convex_hull_order, _close_polygon
    fig, ax = plt.subplots(figsize=(7, 7))

    for ffor, color, label in [
        (base_ffor, "steelblue", "Baseline FFOR"),
        (opt_ffor,  "tomato",    f"FFOR + line ({best_pair[0]}<->{best_pair[1]})"),
    ]:
        if ffor is None or len(ffor.P_flex) < 3: continue
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
    if save_path: fig.savefig(save_path, dpi=150)
    else: plt.show()
    plt.close(fig)