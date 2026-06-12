"""
devices.py
"""

import numpy as np
import gurobipy as gp

def _polygon_vertices(S_max: float, n_sides: int = 8):
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    P = S_max * np.cos(angles)
    Q = S_max * np.sin(angles)
    return P, Q

def _add_polygon_constraint(m: gp.Model, P_var, Q_var, S_max: float, n_sides: int = 8, name_prefix: str = ""):
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    delta = np.pi / n_sides
    rhs = S_max * np.cos(delta)
    for i in range(n_sides):
        a_mid = angles[i] + delta
        nx, ny = np.cos(a_mid), np.sin(a_mid)
        m.addConstr(nx * P_var + ny * Q_var <= rhs, name=f"{name_prefix}_edge_{i}")

class BESSModel:
    def __init__(self, bus: int, S_max_MW: float, E_cap_MWh: float, soc_min: float = 0.2, soc_max: float = 0.9, soc_init: float = 0.5):
        self.bus = bus
        self.S_max = S_max_MW
        self.E_cap = E_cap_MWh
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.soc_init = soc_init

    def add_single_timestep_constraints(self, m: gp.Model, P_var, Q_var, dt_h: float = 0.25, name: str = "bess"):
        _add_polygon_constraint(m, P_var, Q_var, self.S_max, n_sides=8, name_prefix=name)
        m.addConstr(P_var <=  self.S_max, name=f"{name}_P_upper")
        m.addConstr(P_var >= -self.S_max, name=f"{name}_P_lower")
        soc_avail = (self.soc_init - self.soc_min) * self.E_cap
        m.addConstr(P_var <= soc_avail / dt_h, name=f"{name}_soc_discharge")
        soc_room  = (self.soc_max - self.soc_init) * self.E_cap
        m.addConstr(P_var >= -soc_room / dt_h, name=f"{name}_soc_charge")

    def add_multi_timestep_constraints(self, m: gp.Model, P_vars: list, Q_vars: list, soc_vars: list, dt_h: float = 0.25, name: str = "bess"):
        T = len(P_vars)
        m.addConstr(soc_vars[0] == self.soc_init, name=f"{name}_soc0")
        for t in range(T):
            _add_polygon_constraint(m, P_vars[t], Q_vars[t], self.S_max, n_sides=8, name_prefix=f"{name}_t{t}")
            m.addConstr(soc_vars[t] >= self.soc_min, name=f"{name}_socmin_t{t}")
            m.addConstr(soc_vars[t] <= self.soc_max, name=f"{name}_socmax_t{t}")
            if t < T - 1:
                m.addConstr(soc_vars[t + 1] == soc_vars[t] - dt_h * P_vars[t] / self.E_cap, name=f"{name}_soc_dyn_t{t}")

    def for_polygon(self, n_sides: int = 8):
        return _polygon_vertices(self.S_max, n_sides)

class PVModel:
    def __init__(self, bus: int, S_max_MW: float, pf_min: float = 0.9):
        self.bus = bus
        self.S_max = S_max_MW
        self.pf_min = pf_min
        self.tan_phi = np.tan(np.arccos(pf_min))

    def p_max(self, irradiance: float) -> float:
        return self.S_max * irradiance / 1000.0

    def add_constraints(self, m: gp.Model, P_var, Q_var, irradiance: float, name: str = "pv"):
        pmax = self.p_max(irradiance)
        m.addConstr(P_var >= 0,    name=f"{name}_P_lower")
        m.addConstr(P_var <= pmax, name=f"{name}_P_upper")
        m.addConstr(Q_var <=  self.tan_phi * P_var, name=f"{name}_Q_upper")
        m.addConstr(Q_var >= -self.tan_phi * P_var, name=f"{name}_Q_lower")

    def for_polygon(self, irradiance: float = 1000.0):
        pmax = self.p_max(irradiance)
        P = [0, pmax, pmax, 0]
        Q = [0,  self.tan_phi * pmax, -self.tan_phi * pmax, 0]
        return np.array(P), np.array(Q)

class ControllableLoadModel:
    def __init__(self, bus: int, P_max_MW: float, pf: float = 0.95, device_type: str = "hp", T_min: float = 19.0, T_max: float = 23.0, T_init: float = 21.0, q_heat: float = 2.0):
        self.bus = bus
        self.P_max = P_max_MW
        self.pf = pf
        self.tan_phi = np.tan(np.arccos(pf))
        self.type = device_type
        self.T_min = T_min
        self.T_max = T_max
        self.T_init = T_init
        self.q_heat = q_heat

    def add_single_timestep_constraints(self, m: gp.Model, P_var, Q_var, P_base: float = 0.0, name: str = "load"):
        m.addConstr(P_var >= 0,        name=f"{name}_P_lower")
        m.addConstr(P_var <= self.P_max, name=f"{name}_P_upper")
        m.addConstr(Q_var == self.tan_phi * P_var, name=f"{name}_pf")

    def add_multi_timestep_constraints(self, m: gp.Model, P_vars: list, Q_vars: list, T_vars: list, P_base_series: np.ndarray, q_heat_series: np.ndarray, dt_h: float = 0.25, name: str = "load"):
        T = len(P_vars)
        if self.type == "hp":
            m.addConstr(T_vars[0] == self.T_init, name=f"{name}_T0")
        for t in range(T):
            m.addConstr(P_vars[t] >= 0,           name=f"{name}_P_lower_t{t}")
            m.addConstr(P_vars[t] <= self.P_max,  name=f"{name}_P_upper_t{t}")
            m.addConstr(Q_vars[t] == self.tan_phi * P_vars[t], name=f"{name}_pf_t{t}")
            if self.type == "hp" and t < T - 1:
                P_base = P_base_series[t]
                if P_base > 1e-3:
                    P_flex = P_vars[t] - P_base
                    m.addConstr(T_vars[t + 1] == T_vars[t] + (P_flex / P_base) * q_heat_series[t] * dt_h, name=f"{name}_T_dyn_t{t}")
                else:
                    m.addConstr(T_vars[t + 1] == T_vars[t], name=f"{name}_T_dyn_t{t}")
                m.addConstr(T_vars[t] >= self.T_min, name=f"{name}_Tmin_t{t}")
                m.addConstr(T_vars[t] <= self.T_max, name=f"{name}_Tmax_t{t}")

    def for_polygon(self, P_base: float = 0.0):
        P = np.array([0.0, self.P_max])
        Q = self.tan_phi * P
        return P, Q