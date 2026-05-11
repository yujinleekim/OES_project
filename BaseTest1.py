"""
Power Balance Optimisation with Network Line Constraints
=========================================================

Network topology (undirected connections):
    g1  -- d1
    g2  -- d1
    d1  -- d2
    d2  -- d3
    d2  -- d4
    d4  -- d5
    g3  -- d5

All other node pairs have NO connection (zero power flow enforced implicitly
by the absence of a flow variable for that edge).

For each edge a directed flow variable f[edge, t] is introduced.
Positive flow = power flowing in the "forward" direction (as listed above).
Negative flow = power flowing in the "reverse" direction.
The magnitude is bounded by the line capacity: -cap <= f <= +cap.

Nodal power balance (Kirchhoff's Current Law) is enforced at every node:
    sum(injections) + sum(flows in) - sum(flows out) = 0

Generation limits:
    0 <= Pg[g, t] <= 100  for g in {1, 2, 3}

Load profile (MW):
    Pd1 = [20, 20, 20]
    Pd2 = [20, 30, 20]
    Pd3 = [20, 20, 20]
    Pd4 = [20, 20, 20]
    Pd5 = [20, 20, 20]

Objective: Minimise total generation.
"""

import gurobipy as gp
from gurobipy import GRB

# ===========================================================================
# 1.  DATA
# ===========================================================================

T = 3  # number of time steps

# Load profiles  {demand_node: [t0, t1, t2]}
Pd = {
    "d1": [20, 20, 20],
    "d2": [20, 30, 20],
    "d3": [20, 20, 20],
    "d4": [20, 20, 20],
    "d5": [20, 20, 20],
}

# Generator limits  {gen_node: (min_MW, max_MW)}
gen_limits = {
    "g1": (0, 100),
    "g2": (0, 100),
    "g3": (0, 100),
}

# ---------------------------------------------------------------------------
# Network edges  (from_node, to_node) : line_capacity_MW
# ---------------------------------------------------------------------------
# Edit the capacity values here to customise each line independently.
# Only these edges exist; all other node pairs carry zero flow.
# ---------------------------------------------------------------------------
line_capacity = {
    ("g1", "d1"): 100,
    ("g2", "d1"): 100,
    ("d1", "d2"): 100,
    ("d2", "d3"): 100,
    ("d2", "d4"): 100,
    ("d4", "d5"): 100,
    ("g3", "d5"): 100,
}

# Derive the set of all nodes from the edge list + generators + demands
all_nodes = set()
for (u, v) in line_capacity:
    all_nodes.update([u, v])

# ===========================================================================
# 2.  MODEL
# ===========================================================================

model = gp.Model("PowerBalance_Network")

# ---------------------------------------------------------------------------
# Decision variables
# ---------------------------------------------------------------------------

# Generator output  Pg[gen_node, t]
Pg = {}
for g, (lo, hi) in gen_limits.items():
    for t in range(T):
        Pg[g, t] = model.addVar(lb=lo, ub=hi, name=f"Pg_{g}_t{t+1}")

# Line flows  f[edge, t]   (can be negative = reverse flow)
# Each edge (u,v) has one signed flow variable; positive means u→v.
f = {}
for edge, cap in line_capacity.items():
    for t in range(T):
        f[edge, t] = model.addVar(lb=-cap, ub=cap,
                                  name=f"f_{edge[0]}_{edge[1]}_t{t+1}")

# ---------------------------------------------------------------------------
# Objective  –  Minimise total generation
# ---------------------------------------------------------------------------
model.setObjective(
    gp.quicksum(Pg[g, t] for g in gen_limits for t in range(T)),
    GRB.MINIMIZE
)

# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

# Build adjacency lookup: for each node, which edges touch it and in
# which direction?
def flow_into_node(node, t):
    """Returns a Gurobi linear expression for net power flowing INTO node."""
    expr = gp.LinExpr()
    for edge in line_capacity:
        u, v = edge
        if v == node:          # edge arrives at node  →  positive f means inflow
            expr += f[edge, t]
        elif u == node:        # edge leaves from node →  positive f means outflow
            expr -= f[edge, t]
    return expr

for t in range(T):
    for node in all_nodes:
        # Power injected at this node (generation minus demand)
        injection = gp.LinExpr()

        if node in gen_limits:
            injection += Pg[node, t]          # generator injects power

        if node in Pd:
            injection -= Pd[node][t]          # demand withdraws power

        # KCL: net injection == net outflow
        model.addConstr(
            injection + flow_into_node(node, t) == 0,
            name=f"KCL_{node}_t{t+1}"
        )

# ===========================================================================
# 3.  SOLVE
# ===========================================================================

model.optimize()

# ===========================================================================
# 4.  RESULTS
# ===========================================================================

if model.Status == GRB.OPTIMAL:
    print("\n" + "=" * 60)
    print("  OPTIMAL SOLUTION FOUND")
    print("=" * 60)

    for t in range(T):
        total_demand = sum(Pd[d][t] for d in Pd)
        total_gen    = sum(Pg[g, t].X for g in gen_limits)

        print(f"\n{'─'*60}")
        print(f"  Time Step {t+1}   |  Total demand: {total_demand:.1f} MW"
              f"   |  Total gen: {total_gen:.4f} MW")
        print(f"{'─'*60}")

        print("  Generator outputs:")
        for g in gen_limits:
            print(f"    {g:4s} : {Pg[g, t].X:8.4f} MW")

        print("  Line flows  (+ = forward direction as defined above):")
        for edge in line_capacity:
            flow_val = f[edge, t].X
            cap      = line_capacity[edge]
            util     = abs(flow_val) / cap * 100
            print(f"    {edge[0]:3s} → {edge[1]:3s} : {flow_val:8.4f} MW"
                  f"   (capacity {cap} MW, utilisation {util:.1f}%)")

    print(f"\n  Total objective (sum of generation): {model.ObjVal:.4f} MW")
    print("=" * 60)

elif model.Status == GRB.INFEASIBLE:
    print("\nModel is INFEASIBLE.")
    print("Possible causes: line capacities too tight, or demand unreachable.")
    model.computeIIS()
    model.write("infeasible.ilp")
    print("IIS written to infeasible.ilp for debugging.")
else:
    print(f"Optimisation ended with status code: {model.Status}")