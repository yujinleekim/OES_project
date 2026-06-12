"""
ffor_optimizer.py
"""

from dataclasses import dataclass
import numpy as np
import gurobipy as gp
from gurobipy import GRB

@dataclass
class FFORResult:
    P_flex: np.ndarray
    Q_flex: np.ndarray

class FFOROptimizer:

    def __init__(self, grid, bess_list, pv_list, load_list):
        self.grid = grid
        self.bess_list = bess_list
        self.pv_list = pv_list
        self.load_list = load_list
        J_red = self._reduced_jacobian()
        self._J_inv = np.linalg.pinv(J_red)
        self._n_pq = grid.n_buses - 1
        s = grid.pcc_pos
        self._non_pcc = [i for i in range(grid.n_buses) if i != s]
        self._H_branch = self._build_branch_sensitivity()
        self._S_max = np.abs(grid.line_ratings_mva)

    def _reduced_jacobian(self):
        n, s = self.grid.n_buses, self.grid.pcc_pos
        keep = [i for i in range(n) if i != s] + [n + i for i in range(n) if i != s]
        return self.grid.J[np.ix_(keep, keep)]

    def _build_branch_sensitivity(self) -> np.ndarray:
        line_df, bus_df = self.grid.line_df, self.grid.bus_df
        n_pq, n_lines = self._n_pq, len(line_df)
        pos_to_k = np.full(self.grid.n_buses, -1, dtype=int)
        for k, pos in enumerate(self._non_pcc): pos_to_k[pos] = k

        H = np.zeros((2 * n_lines, 2 * n_pq))
        for l, (_, row) in enumerate(line_df.iterrows()):
            fb = bus_df.index.get_loc(int(row['from_bus']))
            tb = bus_df.index.get_loc(int(row['to_bus']))
            Z_base_ohm = (0.4 ** 2) / 1.0
            z = (complex(float(row['r_ohm_per_km']), float(row['x_ohm_per_km'])) * float(row['length_km'])) / Z_base_ohm
            if abs(z) < 1e-12: continue
            y = 1.0 / z
            g_l, b_l = y.real, y.imag
            fi, ti = pos_to_k[fb], pos_to_k[tb]

            if fi >= 0:
                H[l] += -b_l * self._J_inv[fi]; H[l] += g_l * self._J_inv[n_pq + fi]
                H[n_lines + l] += -g_l * self._J_inv[fi]; H[n_lines + l] += -b_l * self._J_inv[n_pq + fi]
            if ti >= 0:
                H[l] += b_l * self._J_inv[ti]; H[l] += -g_l * self._J_inv[n_pq + ti]
                H[n_lines + l] += g_l * self._J_inv[ti]; H[n_lines + l] += b_l * self._J_inv[n_pq + ti]
        return H

    def _add_line_constraints(self, m: gp.Model, dP: dict, dQ: dict, tag: str = ""):
        n_lines, n_pq, n_sides = len(self.grid.line_df), self._n_pq, 8
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        delta = np.pi / n_sides
        inj = ([dP[pos] for pos in self._non_pcc] + [dQ[pos] for pos in self._non_pcc])

        for l in range(n_lines):
            s_l = float(self._S_max[l])
            if s_l <= 1e-4: continue
            
            dPl = gp.quicksum(float(self._H_branch[l, j]) * inj[j] for j in range(2 * n_pq))
            dQl = gp.quicksum(float(self._H_branch[n_lines + l, j]) * inj[j] for j in range(2 * n_pq))
            
            rhs = s_l * np.cos(delta)
            for k in range(n_sides):
                a_mid = angles[k] + delta
                nx, ny = np.cos(a_mid), np.sin(a_mid)
                m.addConstr(nx * dPl + ny * dQl <= rhs, name=f"Ln{l}e{k}{tag}")

    def _irr(self, baseline, t):
        return float(baseline.iloc[min(t, len(baseline) - 1)].get("irradiance", 0.0))

    def _injection_deviations(self, bess_P, bess_Q, pv_P, pv_Q, load_P, load_Q, baseline, t_abs):
        dP = {pos: gp.LinExpr() for pos in self._non_pcc}
        dQ = {pos: gp.LinExpr() for pos in self._non_pcc}

        for i, bess in enumerate(self.bess_list):
            pos = self.grid.bus_pos(bess.bus)
            dP[pos] += bess_P[i]; dQ[pos] += bess_Q[i]
        for i, pv in enumerate(self.pv_list):
            pos = self.grid.bus_pos(pv.bus)
            dP[pos] += pv_P[i]; dQ[pos] += pv_Q[i]
        for i, ld in enumerate(self.load_list):
            pos = self.grid.bus_pos(ld.bus)
            dP[pos] += -load_P[i]; dQ[pos] += -load_Q[i]
        return dP, dQ

    def _add_pcc_balance(self, m, dP, dQ, dP_pcc, dQ_pcc, tag=""):
        m.addConstr(dP_pcc == -gp.quicksum(dP[pos] for pos in self._non_pcc), name=f"Pbal{tag}")
        m.addConstr(dQ_pcc == -gp.quicksum(dQ[pos] for pos in self._non_pcc), name=f"Qbal{tag}")

    def _add_voltage_constraints(self, m, dP, dQ, tag=""):
        n_pq = self._n_pq
        v_min, v_max = self.grid.v_limits
        lo, hi = v_min - 1.0, v_max - 1.0
        inj = ([dP[pos] for pos in self._non_pcc] + [dQ[pos] for pos in self._non_pcc])
        J_U = self._J_inv[n_pq:]

        for b in range(n_pq):
            row = J_U[b]
            expr = gp.quicksum(float(row[j]) * inj[j] for j in range(2 * n_pq) if abs(row[j]) > 1e-8)
            m.addConstr(expr <= hi, name=f"Vhi{b}{tag}")
            m.addConstr(expr >= lo, name=f"Vlo{b}{tag}")

    def _build_single_lp(self, t_idx, baseline, cos_th, sin_th, enforce_voltage=True):
        m = gp.Model()
        m.setParam("OutputFlag", 0)
        m.setParam("NumericFocus", 3)
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
        
        if enforce_voltage:
            self._add_voltage_constraints(m, dP, dQ)
            self._add_line_constraints(m, dP, dQ)

        m.update()
        m.setObjective(cos_th * dP_pcc + sin_th * dQ_pcc, GRB.MAXIMIZE)
        m.optimize()

        if m.Status == GRB.OPTIMAL: 
            return dP_pcc.X, dQ_pcc.X
            
        return np.nan, np.nan

    def compute_ffor_single_timestep(self, t_idx, baseline, n_directions: int = 36) -> FFORResult:
        angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
        pts = [self._build_single_lp(t_idx, baseline, np.cos(a), np.sin(a), enforce_voltage=True) for a in angles]
        pts = [(p, q) for p, q in pts if not np.isnan(p)]
        return FFORResult(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))

    def _build_multi_lp(self, t_start, T, baseline, dt_h, cos_th, sin_th, enforce_voltage=True):
        m = gp.Model()
        m.setParam("OutputFlag", 0)
        m.setParam("NumericFocus", 3)

        bess_Pall, bess_Qall = [], []
        for i, bess in enumerate(self.bess_list):
            Pv = [m.addVar(lb=-bess.S_max, ub=bess.S_max, name=f"b{i}P{t}") for t in range(T)]
            Qv = [m.addVar(lb=-bess.S_max, ub=bess.S_max, name=f"b{i}Q{t}") for t in range(T)]
            sv = [m.addVar(lb=bess.soc_min, ub=bess.soc_max, name=f"b{i}s{t}") for t in range(T)]
            bess.add_multi_timestep_constraints(m, Pv, Qv, sv, dt_h=dt_h, name=f"b{i}")
            bess_Pall.append(Pv); bess_Qall.append(Qv)

        pv_Pall, pv_Qall = [], []
        for i, pv in enumerate(self.pv_list):
            Pv, Qv = [], []
            for t in range(T):
                irr = self._irr(baseline, t_start + t)
                p = m.addVar(lb=0, ub=pv.p_max(irr), name=f"pv{i}P{t}")
                q = m.addVar(lb=-pv.S_max * pv.tan_phi, ub=pv.S_max * pv.tan_phi, name=f"pv{i}Q{t}")
                pv.add_constraints(m, p, q, irradiance=irr, name=f"pv{i}t{t}")
                Pv.append(p); Qv.append(q)
            pv_Pall.append(Pv); pv_Qall.append(Qv)

        load_Pall, load_Qall = [], []
        for i, ld in enumerate(self.load_list):
            Pv, Qv = [], []
            for t in range(T):
                p = m.addVar(lb=0, ub=ld.P_max, name=f"l{i}P{t}")
                q = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"l{i}Q{t}")
                Pv.append(p); Qv.append(q)
            if ld.type == "hp":
                T_vars = [m.addVar(lb=ld.T_min, ub=ld.T_max, name=f"l{i}T{t}") for t in range(T)]
                col = f"bus_{ld.bus}_P_hp_base"
                P_base_series = np.array([
                    float(baseline.iloc[min(t_start + t, len(baseline) - 1)].get(col, 0.0))
                    for t in range(T)
                ])
                ld.add_multi_timestep_constraints(
                    m, Pv, Qv, T_vars, P_base_series,
                    np.full(T, ld.q_heat), dt_h=dt_h, name=f"l{i}"
                )
            else:
                for t in range(T):
                    ld.add_single_timestep_constraints(m, Pv[t], Qv[t], P_base=0.0, name=f"l{i}t{t}")
            load_Pall.append(Pv); load_Qall.append(Qv)

        m.update()
        dP_pcc = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="dP_pcc")
        dQ_pcc = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="dQ_pcc")

        for t in range(T):
            t_abs = t_start + t
            dP, dQ = self._injection_deviations(
                [bess_Pall[i][t] for i in range(len(self.bess_list))],
                [bess_Qall[i][t] for i in range(len(self.bess_list))],
                [pv_Pall[i][t]   for i in range(len(self.pv_list))],
                [pv_Qall[i][t]   for i in range(len(self.pv_list))],
                [load_Pall[i][t] for i in range(len(self.load_list))],
                [load_Qall[i][t] for i in range(len(self.load_list))],
                baseline, t_abs,
            )
            self._add_pcc_balance(m, dP, dQ, dP_pcc, dQ_pcc, tag=str(t))
            if enforce_voltage:
                self._add_voltage_constraints(m, dP, dQ, tag=f"t{t}")
                self._add_line_constraints(m, dP, dQ, tag=f"t{t}")

        m.update()
        m.setObjective(cos_th * dP_pcc + sin_th * dQ_pcc, GRB.MAXIMIZE)
        m.optimize()

        if m.Status == GRB.OPTIMAL: return dP_pcc.X, dQ_pcc.X
        return np.nan, np.nan

    def compute_ffor_multi_timestep(self, t_start, duration_h, baseline, n_directions: int = 36) -> FFORResult:
        dt_h = 0.25
        T = max(1, round(duration_h / dt_h))
        angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
        pts = [self._build_multi_lp(t_start, T, baseline, dt_h, np.cos(a), np.sin(a), enforce_voltage=True) for a in angles]
        pts = [(p, q) for p, q in pts if not np.isnan(p)]
        return FFORResult(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))