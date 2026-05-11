"""
Feasible Operating Regions  –  (Pg1,Pg2), (Pg1,Pg3) and (Pg2,Pg3)
===================================================================
For each time step and each generator pair, this script sweeps the
x-axis generator across its feasible range, fixes it as a parameter,
and solves the network-constrained dispatch to find the maximum and
minimum feasible value of the y-axis generator.

The resulting frontier is plotted as a shaded feasible region.
Layout: 3 rows (one per pair) x 3 columns (one per time step).

Network topology:
    g1 -- d1 -- d2 -- d3
    g2 -- d1    d2 -- d4 -- d5 -- g3

Load profile (MW):
    Pd1 = [20, 20, 20]
    Pd2 = [20, 20, 20]
    Pd3 = [20, 30, 40]   <- ramping load
    Pd4 = [20, 20, 20]
    Pd5 = [20, 20, 20]

Line capacities (all 50 MW – edit line_capacity dict to customise):
    g1->d1, g2->d1, d1->d2, d2->d3, d2->d4, d4->d5, g3->d5 : 50 MW each
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gurobipy as gp
from gurobipy import GRB

# ===========================================================================
# 1.  DATA
# ===========================================================================

T = 3

Pd = {
    "d1": [20, 20, 20],
    "d2": [20, 20, 20],
    "d3": [20, 30, 40],
    "d4": [20, 20, 20],
    "d5": [20, 20, 20],
}

gen_limits = {
    "g1": (0, 100),
    "g2": (0, 100),
    "g3": (0, 100),
}

# ---------------------------------------------------------------------------
# Edit capacities here – each line is independently customisable
# ---------------------------------------------------------------------------
line_capacity = {
    ("g1", "d1"): 50,
    ("g2", "d1"): 50,
    ("d1", "d2"): 50,
    ("d2", "d3"): 50,
    ("d2", "d4"): 50,
    ("d4", "d5"): 50,
    ("g3", "d5"): 50,
}

all_nodes = set()
for (u, v) in line_capacity:
    all_nodes.update([u, v])

# ===========================================================================
# 2.  GENERIC HELPER – fix one generator, optimise bound on another
# ===========================================================================

def solve_bound(fixed_gen: str, fixed_val: float,
                opt_gen: str, t: int,
                minimise: bool) -> float | None:
    """
    Fix `fixed_gen` = fixed_val at time step t.
    Minimise or maximise `opt_gen` subject to all network + KCL constraints.
    Returns the optimal value of opt_gen, or None if infeasible.
    """
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(env=env)

    Pg = {}
    for g, (lo, hi) in gen_limits.items():
        if g == fixed_gen:
            Pg[g, t] = m.addVar(lb=fixed_val, ub=fixed_val, name=f"Pg_{g}_t{t+1}")
        else:
            Pg[g, t] = m.addVar(lb=lo, ub=hi, name=f"Pg_{g}_t{t+1}")

    f = {}
    for edge, cap in line_capacity.items():
        f[edge, t] = m.addVar(lb=-cap, ub=cap,
                               name=f"f_{edge[0]}_{edge[1]}_t{t+1}")

    m.setObjective(Pg[opt_gen, t], GRB.MINIMIZE if minimise else GRB.MAXIMIZE)

    def flow_into(node):
        expr = gp.LinExpr()
        for edge in line_capacity:
            u, v = edge
            if v == node:
                expr += f[edge, t]
            elif u == node:
                expr -= f[edge, t]
        return expr

    for node in all_nodes:
        injection = gp.LinExpr()
        if node in gen_limits:
            injection += Pg[node, t]
        if node in Pd:
            injection -= Pd[node][t]
        m.addConstr(injection + flow_into(node) == 0,
                    name=f"KCL_{node}_t{t+1}")

    m.optimize()

    if m.Status == GRB.OPTIMAL:
        return Pg[opt_gen, t].X
    return None


# ===========================================================================
# 3.  COMPUTE FEASIBLE REGION for a given (x_gen, y_gen) pair
# ===========================================================================

def compute_region(x_gen: str, y_gen: str, t: int, n_points: int = 200):
    """
    Sweep x_gen from 0 to its max, find (y_min, y_max) at each point.
    Returns (x_vals, y_lo_vals, y_hi_vals).
    """
    x_max = gen_limits[x_gen][1]
    sweep = np.linspace(0, x_max, n_points)

    x_vals, y_lo_vals, y_hi_vals = [], [], []

    for xv in sweep:
        y_min = solve_bound(x_gen, xv, y_gen, t, minimise=True)
        y_max = solve_bound(x_gen, xv, y_gen, t, minimise=False)

        if y_min is not None and y_max is not None and y_max >= y_min - 1e-6:
            x_vals.append(xv)
            y_lo_vals.append(max(0.0, y_min))
            y_hi_vals.append(min(100.0, y_max))

    return x_vals, y_lo_vals, y_hi_vals


# ===========================================================================
# 4.  PLOT  –  3 rows (pairs) x 3 columns (time steps)
# ===========================================================================

PAIRS = [
    ("g1", "g2", "Pg1 (MW)", "Pg2 (MW)"),
    ("g1", "g3", "Pg1 (MW)", "Pg3 (MW)"),
    ("g2", "g3", "Pg2 (MW)", "Pg3 (MW)"),
]

total_demand = [sum(Pd[d][t] for d in Pd) for t in range(T)]
LABELS = [f"Time step {t+1}  (demand = {total_demand[t]} MW)" for t in range(T)]
COLORS = ["#3B6D11", "#185FA5", "#993556"]
FILLS  = ["#C0DD97", "#B5D4F4", "#F4C0D1"]
N_POINTS = 200

print("Computing feasible regions – this may take a moment...")
print(f"  Total demand per step: {total_demand}\n")

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle(
    "Feasible Operating Regions: (Pg1,Pg2), (Pg1,Pg3) and (Pg2,Pg3)\n"
    "Load profile: Pd3 = [20, 30, 40] MW  |  50 MW line caps",
    fontsize=13, fontweight="bold"
)

for row, (x_gen, y_gen, x_lbl, y_lbl) in enumerate(PAIRS):
    print(f"--- Pair: ({x_gen}, {y_gen}) ---")
    for t in range(T):
        ax = axes[row, t]

        x_vals, y_lo, y_hi = compute_region(x_gen, y_gen, t, N_POINTS)

        if x_vals:
            ax.fill_between(x_vals, y_lo, y_hi,
                            color=FILLS[t], alpha=0.7, label="Feasible region")
            ax.plot(x_vals, y_hi, color=COLORS[t], linewidth=1.8,
                    label="Upper boundary")
            ax.plot(x_vals, y_lo, color=COLORS[t], linewidth=1.8,
                    linestyle="--", label="Lower boundary")
            print(f"  t={t+1}: {x_gen} feasible in [{min(x_vals):.1f}, {max(x_vals):.1f}] MW")
        else:
            ax.text(50, 50, "Infeasible", ha="center", va="center",
                    fontsize=12, color="red", fontweight="bold")
            print(f"  t={t+1}: INFEASIBLE")

        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.set_xlabel(x_lbl, fontsize=11)
        if t == 0:
            ax.set_ylabel(y_lbl, fontsize=11)
        ax.set_title(LABELS[t], fontsize=10, color=COLORS[t], fontweight="bold")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_aspect("equal")

        patch  = mpatches.Patch(color=FILLS[t], alpha=0.7, label="Feasible region")
        upper, = ax.plot([], [], color=COLORS[t], linewidth=1.8,
                         label="Upper boundary")
        lower, = ax.plot([], [], color=COLORS[t], linewidth=1.8,
                         linestyle="--", label="Lower boundary")
        ax.legend(handles=[patch, upper, lower], fontsize=8, loc="upper right")

    print()

plt.tight_layout()
outfile = "feasible_region_all_pairs.png"
plt.savefig(outfile, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved to {outfile}")