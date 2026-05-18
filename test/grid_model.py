"""
grid_model.py
─────────────
Wasterkingen LV distribution grid (Swiss-PDGs dataset, grid ID: 70-1_0_4).
  - Loads real MATPOWER CSV data; falls back to synthetic 9-bus MV network
  - Builds the linearised power-flow Jacobian  J = [JPθ JPU; JQθ JQU]
  - Auto-derives BESS / PV / controllable-load placement from the network
  - Generates synthetic baseline time series scaled to actual load level
"""

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.io_utils import from_dict_of_dfs
from pathlib import Path
from devices import BESSModel, PVModel, ControllableLoadModel


# ── paths to real Wasterkingen data ───────────────────────────────────────────

_HERE          = Path(__file__).parent
_REAL_MPC_DIR  = _HERE.parent / "grids/matpower_data/LV/Midlands-Periurban/70-1_0_4"
_REAL_PP_FILE  = _HERE.parent / "70-1_0_4_grid.xlsx"


# ─────────────────────────────────────────────────────────────────────────────
# Jacobian helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_jacobian(net):
    """
    Linearised power-flow Jacobian around the current operating point.

        J = | JPθ  JPU |
            | JQθ  JQU |

    Row/column order follows net.bus.index (positional order).
    Uses converged power-flow voltages/angles from net.res_bus when available;
    falls back to flat start (|V|=1 p.u., θ=0) otherwise.

    Fully vectorised — no Python-level double loop.
    """
    n = len(net.bus)

    # ── operating point ───────────────────────────────────────────
    try:
        _vm = net.res_bus.vm_pu.values
        _va = net.res_bus.va_degree.values
        if len(_vm) == n and not np.any(np.isnan(_vm)):
            V_pu   = _vm.astype(float)
            th_rad = np.deg2rad(_va.astype(float))
        else:
            raise ValueError("res_bus size mismatch or NaN")
    except Exception:
        V_pu   = np.ones(n)
        th_rad = np.zeros(n)

    # ── admittance matrix ─────────────────────────────────────────
    g    = np.zeros((n, n))
    b    = np.zeros((n, n))

    bus_idx = {idx: pos for pos, idx in enumerate(net.bus.index)}
    for _, row in net.line.iterrows():
        i = bus_idx[int(row.from_bus)]
        k = bus_idx[int(row.to_bus)]
        z = complex(float(row.r_ohm_per_km), float(row.x_ohm_per_km)) * float(row.length_km)
        if abs(z) < 1e-12:
            continue
        y        = 1.0 / z
        gik, bik = y.real, y.imag
        g[i, k] -= gik;  g[k, i] -= gik
        g[i, i] += gik;  g[k, k] += gik
        b[i, k] -= bik;  b[k, i] -= bik
        b[i, i] += bik;  b[k, k] += bik

    # ── vectorised Jacobian ───────────────────────────────────────
    Vi    = V_pu[:, None]          # (n, 1)
    Vk    = V_pu[None, :]          # (1, n)
    th_ik = th_rad[:, None] - th_rad[None, :]   # (n, n)

    sin_ik = np.sin(th_ik)
    cos_ik = np.cos(th_ik)

    # Off-diagonal elements (correct at all i≠k; diagonal overwritten below)
    JPt =  Vi * Vk * ( g * sin_ik - b * cos_ik)
    JPU =  Vi *      ( g * cos_ik + b * sin_ik)
    JQt =  Vi * Vk * (-g * cos_ik - b * sin_ik)
    JQU =  Vi *      ( g * sin_ik - b * cos_ik)

    # Diagonal elements (derived from off-diagonal row sums)
    # JPt[i,i] = -Vi * sum_{j≠i} Vj*(g_ij*sin_ij - b_ij*cos_ij)
    #           = sum_{j≠i} JPt[i,j]  but with negated sign and j=i zeroed
    diag_mask = np.eye(n, dtype=bool)

    JPt_offdiag_sum = JPt.copy(); np.fill_diagonal(JPt_offdiag_sum, 0)
    JQU_offdiag_sum = JQU.copy(); np.fill_diagonal(JQU_offdiag_sum, 0)
    JPU_offdiag_sum = JPU.copy(); np.fill_diagonal(JPU_offdiag_sum, 0)
    JQt_offdiag_sum = JQt.copy(); np.fill_diagonal(JQt_offdiag_sum, 0)

    np.fill_diagonal(JPt, -JPt_offdiag_sum.sum(axis=1))
    np.fill_diagonal(JQU, -JQU_offdiag_sum.sum(axis=1) + 2 * V_pu * np.diag(g) * 0
                          + V_pu * np.diag(b) * 0)   # simplified below
    np.fill_diagonal(JPU,  JPU_offdiag_sum.sum(axis=1) + V_pu * np.diag(g))
    np.fill_diagonal(JQt, -JQt_offdiag_sum.sum(axis=1))

    # JQU diagonal: Vi*(sum_{j≠i} g_ij) + 2*Vi*g_ii  →  Vi*sum_j g_ij + Vi*g_ii
    # Equivalently: row-sum of off-diagonal JQU + 2*Vi²*g_ii (self-admittance)
    np.fill_diagonal(JQU,
                     -JQU_offdiag_sum.sum(axis=1)
                     + 2.0 * V_pu * np.diag(b))

    J = np.block([[JPt, JPU], [JQt, JQU]])
    return J, {"JPt": JPt, "JPU": JPU, "JQt": JQt, "JQU": JQU}


# ─────────────────────────────────────────────────────────────────────────────
# Grid class
# ─────────────────────────────────────────────────────────────────────────────

class WasterkingenGrid:
    """
    Wasterkingen distribution grid.
    Loads real LV data (70-1_0_4) when available; falls back to a synthetic
    9-bus MV network for offline/demo use.
    """

    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self.net      = self._load_or_build_network()
        self.n_buses  = len(self.net.bus)
        self.n_lines  = len(self.net.line)
        self.pcc_bus  = int(self.net.ext_grid.bus.iloc[0])
        self.J        = None
        self.J_blocks = None

        # Device placement derived from network topology
        self._BESS_DATA, self._PV_DATA, self._LOAD_DATA = self._make_device_data()

    # ── bus index helpers ─────────────────────────────────────────────────────

    def bus_pos(self, bus_idx: int) -> int:
        """Pandapower bus index → row/column position in the Jacobian matrix."""
        return self.net.bus.index.get_loc(bus_idx)

    @property
    def pcc_pos(self) -> int:
        """Positional index of the PCC (slack) bus in the Jacobian."""
        return self.bus_pos(self.pcc_bus)

    # ── network loading ───────────────────────────────────────────────────────

    def _load_or_build_network(self) -> pp.pandapowerNet:
        # 1. Try pandapower Excel (with geodata-compat shim for pp 3.x)
        if _REAL_PP_FILE.exists():
            try:
                net = self._load_from_excel(_REAL_PP_FILE)
                self._repair_dataframes(net)   # repair once at load time
                print(f"  Loaded real grid: {_REAL_PP_FILE.name} "
                      f"({len(net.bus)} buses, {len(net.line)} lines)")
                return net
            except Exception as e:
                print(f"  [warn] Excel load failed ({e}), trying CSV ...")

        # 2. Try MATPOWER CSV
        if _REAL_MPC_DIR.exists():
            try:
                net = self._load_from_matpower_csv(_REAL_MPC_DIR)
                self._repair_dataframes(net)   # repair once at load time
                print(f"  Loaded real grid from MATPOWER CSV "
                      f"({len(net.bus)} buses, {len(net.line)} lines)")
                return net
            except Exception as e:
                print(f"  [warn] MATPOWER CSV load failed ({e}), using synthetic ...")

        # 3. Synthetic fallback
        print("  [warn] Real data not found - using synthetic 9-bus MV network")
        return self._build_synthetic_network()

    @staticmethod
    def _repair_dataframes(net: pp.pandapowerNet) -> None:
        """
        Pandapower's runpp calls .fillna() on DataFrame columns. When a column
        is stored as a raw numpy ndarray (from older Excel exports via
        from_dict_of_dfs), this crashes because numpy.ndarray is an immutable
        C type with no .fillna() method.

        Fix: rebuild every column in every element table as a genuine pd.Series
        by round-tripping through a Python list. Called once at load time so
        it never runs during the hot path (build_jacobian / line_add loop).
        """
        _lm_cols = ("const_z_p_percent", "const_i_p_percent",
                    "const_z_q_percent", "const_i_q_percent")

        tables = ["load", "sgen", "gen", "ward", "xward", "shunt",
                  "storage", "asymmetric_load", "asymmetric_sgen",
                  "line", "trafo", "trafo3w", "ext_grid", "bus"]

        for tbl_name in tables:
            tbl = getattr(net, tbl_name, None)
            if tbl is None or not isinstance(tbl, pd.DataFrame) or tbl.empty:
                continue
            rebuilt = {}
            for col in tbl.columns:
                raw = tbl[col]
                try:
                    vals = raw.tolist() if hasattr(raw, "tolist") else list(raw)
                    rebuilt[col] = pd.Series(vals, index=tbl.index)
                except Exception:
                    rebuilt[col] = pd.Series([0.0] * len(tbl), index=tbl.index)
            setattr(net, tbl_name, pd.DataFrame(rebuilt, index=tbl.index))

        # Ensure load-model columns exist on load/sgen with value 0.0
        for tbl_name in ("load", "sgen"):
            tbl = getattr(net, tbl_name, None)
            if tbl is None or not isinstance(tbl, pd.DataFrame) or tbl.empty:
                continue
            for col in _lm_cols:
                if col not in tbl.columns:
                    tbl[col] = 0.0
                else:
                    tbl[col] = pd.to_numeric(tbl[col], errors="coerce").fillna(0.0)

    def _load_from_excel(self, path: Path) -> pp.pandapowerNet:
        """
        Load pandapower Excel, dropping geodata sheets that cause KeyErrors
        in pandapower 3.x when the empty network lacks those tables.
        """
        xls = pd.read_excel(str(path), sheet_name=None, index_col=0, engine="openpyxl")
        for sheet in ("line_geodata", "bus_geodata"):
            xls.pop(sheet, None)
        return from_dict_of_dfs(xls)

    def _load_from_matpower_csv(self, csv_dir: Path) -> pp.pandapowerNet:
        name      = csv_dir.name
        bus_df    = pd.read_csv(csv_dir / f"{name}_bus_data.csv")
        branch_df = pd.read_csv(csv_dir / f"{name}_branch_data.csv")

        vn_kv  = float(bus_df["baseKV"].iloc[0])   # 0.4 kV for LV
        sn_mva = 100.0                               # MATPOWER default base
        z_base = vn_kv ** 2 / sn_mva               # Ohm

        net = pp.create_empty_network(f_hz=50, sn_mva=sn_mva)

        # Buses
        bus_map = {}
        for _, row in bus_df.iterrows():
            idx = pp.create_bus(net, vn_kv=vn_kv,
                                name=f"Bus_{int(row.BUS_I)}",
                                max_vm_pu=float(row.Vmax),
                                min_vm_pu=float(row.Vmin))
            bus_map[int(row.BUS_I)] = idx

        # External grid (slack bus)
        for _, row in bus_df[bus_df["BUS_TYPE"] == 3].iterrows():
            pp.create_ext_grid(net, bus=bus_map[int(row.BUS_I)],
                               vm_pu=1.0, va_degree=0.0,
                               name="HV/LV Transformer")

        # Loads
        for _, row in bus_df.iterrows():
            if float(row.Pd) > 0 or float(row.Qd) > 0:
                pp.create_load(net, bus=bus_map[int(row.BUS_I)],
                               p_mw=float(row.Pd), q_mvar=float(row.Qd),
                               name=f"Load_{int(row.BUS_I)}")

        # Lines (impedance in p.u. on 100-MVA base → convert to Ohm)
        for _, row in branch_df.iterrows():
            if int(row.BR_STATUS) == 0:
                continue
            r    = float(row.BR_R) * z_base
            x    = float(row.BR_X) * z_base
            b_pu = float(row.BR_B)
            c_nf = b_pu / (2 * np.pi * 50 * z_base) * 1e9 if b_pu > 0 else 0.0
            pp.create_line_from_parameters(
                net,
                from_bus=bus_map[int(row.F_BUS)],
                to_bus=bus_map[int(row.T_BUS)],
                length_km=1.0,          # length absorbed into r/x totals
                r_ohm_per_km=r,
                x_ohm_per_km=x,
                c_nf_per_km=c_nf,
                max_i_ka=0.2,
                name=f"Line_{int(row.F_BUS)}-{int(row.T_BUS)}",
            )

        return net

    def _build_synthetic_network(self) -> pp.pandapowerNet:
        """9-bus MV synthetic fallback (original sketch)."""
        net = pp.create_empty_network(f_hz=50, sn_mva=10)
        vn  = 16.0
        for i in range(9):
            pp.create_bus(net, vn_kv=vn, name=f"Bus {i}")
        pp.create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0.0,
                           name="HV/MV Transformer")
        pp.create_sgen(net, bus=1, p_mw=0.5, q_mvar=0.0, name="Gen1_RoR")
        pp.create_sgen(net, bus=2, p_mw=0.3, q_mvar=0.0, name="Gen2_RoR")
        pp.create_sgen(net, bus=3, p_mw=0.0, q_mvar=0.0, name="BESS")
        pp.create_load(net, bus=4, p_mw=1.0, q_mvar=0.10, name="Load_A")
        pp.create_load(net, bus=5, p_mw=0.7, q_mvar=0.07, name="Load_B")
        pp.create_load(net, bus=6, p_mw=0.5, q_mvar=0.05, name="Load_C")
        pp.create_load(net, bus=7, p_mw=0.8, q_mvar=0.08, name="Load_D")
        pp.create_load(net, bus=8, p_mw=0.4, q_mvar=0.04, name="Load_E")
        for fb, tb, length in [(0,1,2.0),(0,4,3.0),(1,2,1.5),(1,5,2.5),
                                (2,3,1.0),(3,6,2.0),(4,7,1.8),(5,8,2.2)]:
            pp.create_line_from_parameters(
                net, from_bus=fb, to_bus=tb, length_km=length,
                r_ohm_per_km=0.5, x_ohm_per_km=0.4,
                c_nf_per_km=10.0, max_i_ka=0.2,
                name=f"Line_{fb}-{tb}")
        return net

    # ── device placement ──────────────────────────────────────────────────────

    def _make_device_data(self):
        """
        Derive device placement from the loaded network.

        LV real grid: selects top-load buses; scales device ratings to LV level.
        MV synthetic:  returns the original hardcoded values.
        """
        # Use load-bus voltages: real LV grids have loads at < 1 kV even if an
        # MV slack bus exists (e.g. transformer HV side at 20 kV).
        load_vn = self.net.bus.loc[self.net.load.bus.values, "vn_kv"]
        is_lv   = float(load_vn.max()) < 1.0

        if not is_lv:
            bess = [{"bus": 3, "S_max_MW": 4.0, "E_cap_MWh": 5.0,
                     "soc_min": 0.4, "soc_max": 0.8, "soc_init": 0.6}]
            pv   = [{"bus": 4, "S_max_MW": 0.5, "pf_min": 0.9},
                    {"bus": 5, "S_max_MW": 0.3, "pf_min": 0.9},
                    {"bus": 6, "S_max_MW": 0.8, "pf_min": 0.9},
                    {"bus": 7, "S_max_MW": 0.2, "pf_min": 0.9}]
            load = [{"bus": 4, "P_max_MW": 1.2, "pf": 0.95, "device_type": "hp"},
                    {"bus": 5, "P_max_MW": 0.8, "pf": 0.95, "device_type": "hp"},
                    {"bus": 6, "P_max_MW": 0.6, "pf": 0.95, "device_type": "boiler"},
                    {"bus": 7, "P_max_MW": 0.9, "pf": 0.95, "device_type": "hp"},
                    {"bus": 8, "P_max_MW": 0.5, "pf": 0.95, "device_type": "hp"}]
            return bess, pv, load

        # Real LV: sort load buses by peak demand
        load_df = (self.net.load[self.net.load.bus != self.pcc_bus]
                   .copy()
                   .sort_values("p_mw", ascending=False)
                   .reset_index(drop=True))

        # Total network load — size devices relative to full grid capacity
        p_total = max(float(self.net.load.p_mw.sum()), 0.01)

        # BESS: 40 % of total network load
        top_bus = int(load_df.iloc[0].bus)
        bess_s  = round(max(p_total * 0.40, 0.005), 5)
        bess = [{"bus": top_bus,
                 "S_max_MW":  bess_s,
                 "E_cap_MWh": bess_s * 2.0,
                 "soc_min": 0.2, "soc_max": 0.9, "soc_init": 0.5}]

        # PV: 50 % of total network load per unit at top-4 load buses
        pv_s = round(max(p_total * 0.50, 0.003), 5)
        pv = [{"bus": int(load_df.iloc[i].bus),
               "S_max_MW": pv_s,
               "pf_min": 0.9}
              for i in range(min(4, len(load_df)))]

        # Controllable loads: 50 % of total network load per unit, buses 4–8
        load_s = round(max(p_total * 0.50, 0.002), 5)
        load = [{"bus": int(load_df.iloc[i].bus),
                 "P_max_MW": load_s,
                 "pf": 0.95, "device_type": "hp"}
                for i in range(min(4, len(load_df)), min(9, len(load_df)))]

        return bess, pv, load

    # ── Jacobian ──────────────────────────────────────────────────────────────

    def build_jacobian(self):
        # Clear any stale res_bus from a previous (possibly failed) run so
        # that _build_jacobian reliably falls back to flat start on failure.
        self.net.res_bus = pd.DataFrame(
            columns=["vm_pu", "va_degree", "p_mw", "q_mvar"]
        )
        try:
            self._repair_dataframes(self.net)
            pp.runpp(
                self.net,
                algorithm="nr",
                numba=False,
                lightsim2grid=False,
                voltage_depend_loads=False,
            )
            print("      Power flow converged — Jacobian built around operating point")
        except Exception as e:
            print(f"      [warn] Power flow did not converge ({e}) — using flat start")
        self.J, self.J_blocks = _build_jacobian(self.net)
        return self.J

    # ── device accessors ──────────────────────────────────────────────────────

    def get_bess_devices(self):
        return [BESSModel(**d) for d in self._BESS_DATA]

    def get_pv_devices(self):
        return [PVModel(**d) for d in self._PV_DATA]

    def get_controllable_loads(self):
        return [ControllableLoadModel(**d) for d in self._LOAD_DATA]

    # ── baseline time series ──────────────────────────────────────────────────

    def get_baseline(self, date: str = "2021-09-01",
                     resolution_min: int = 15) -> pd.DataFrame:
        csv_file = self.data_dir / f"baseline_{date}.csv"
        if csv_file.exists():
            return pd.read_csv(csv_file, parse_dates=["time"])
        return self._synthetic_baseline(date, resolution_min)

    def _synthetic_baseline(self, date: str, resolution_min: int) -> pd.DataFrame:
        n_steps     = 24 * 60 // resolution_min
        t           = np.linspace(0, 24, n_steps, endpoint=False)
        irradiance  = np.clip(800 * np.exp(-0.5 * ((t - 12) / 3) ** 2), 0, None)
        temperature = 15 + 8 * np.sin(2 * np.pi * (t - 6) / 24)

        rows = []
        for ti, irr, temp in zip(t, irradiance, temperature):
            row = {"time": ti, "irradiance": irr, "temperature": temp}

            # PV baselines
            for d in self._PV_DATA:
                b = d["bus"]
                row[f"bus_{b}_P_pv_base"] = d["S_max_MW"] * irr / 1000.0
                row[f"bus_{b}_Q_pv_base"] = 0.0

            # Load (HP) baselines
            for d in self._LOAD_DATA:
                b       = d["bus"]
                tan_phi = np.tan(np.arccos(d["pf"]))
                if d["device_type"] == "hp":
                    p_hp = d["P_max_MW"] * max(0, (15 - temp) / 23)
                else:
                    p_hp = d["P_max_MW"] * 0.3
                row[f"bus_{b}_P_hp_base"] = p_hp
                row[f"bus_{b}_Q_hp_base"] = p_hp * tan_phi

            rows.append(row)

        return pd.DataFrame(rows)

    # ── line / voltage limits ─────────────────────────────────────────────────

    @property
    def line_ratings_mva(self) -> np.ndarray:
        bus_vn = self.net.bus.vn_kv  # Series indexed by bus id
        from_buses = self.net.line.from_bus.values
        v_kv = np.array([bus_vn.at[b] if b in bus_vn.index else 0.4
                         for b in from_buses], dtype=float)
        return self.net.line.max_i_ka.values * v_kv * np.sqrt(3)

    @property
    def v_limits(self) -> tuple:
        """
        Voltage band for linearised constraints.
        Widened beyond physical limits to absorb flat-start linearisation error.
        """
        load_vn = self.net.bus.loc[self.net.load.bus.values, "vn_kv"]
        is_lv   = float(load_vn.max()) < 1.0
        return (0.85, 1.15) if is_lv else (0.90, 1.10)