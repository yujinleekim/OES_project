"""
grid_model.py
─────────────
Wasterkingen LV distribution grid (Swiss-PDGs dataset, grid ID: 70-1_0_4).
  - Loads real grid data directly from the .xlsx file via Pandas (no Pandapower)
  - Builds the linearised power-flow Jacobian J = [JPθ JPU; JQθ JQU] analytically
  - Auto-derives BESS / PV / controllable-load placement from the network
  - Generates synthetic baseline time series scaled to actual load level
"""

import numpy as np
import pandas as pd
from pathlib import Path
from devices import BESSModel, PVModel, ControllableLoadModel


# ─────────────────────────────────────────────────────────────────────────────
# Jacobian helper (Analytic / Flat Start)
# ─────────────────────────────────────────────────────────────────────────────

def _build_jacobian(bus_df: pd.DataFrame, line_df: pd.DataFrame):
    """
    Linearised power-flow Jacobian evaluated at a flat start (|V|=1.0 p.u., θ=0).

        J = | JPθ  JPU |
            | JQθ  JQU |

    Fully vectorised -- no Python-level double loop.
    """
    n = len(bus_df)

    V_pu   = np.ones(n)
    th_rad = np.zeros(n)

    g = np.zeros((n, n))
    b = np.zeros((n, n))

    bus_idx = {idx: pos for pos, idx in enumerate(bus_df.index)}

    for _, row in line_df.iterrows():
        if 'in_service' in row and not row['in_service']:
            continue
        i = bus_idx[int(row['from_bus'])]
        k = bus_idx[int(row['to_bus'])]
        z = complex(float(row['r_ohm_per_km']), float(row['x_ohm_per_km'])) * float(row['length_km'])
        if abs(z) < 1e-12:
            continue
        y        = 1.0 / z
        gik, bik = y.real, y.imag
        g[i, k] -= gik;  g[k, i] -= gik
        g[i, i] += gik;  g[k, k] += gik
        b[i, k] -= bik;  b[k, i] -= bik
        b[i, i] += bik;  b[k, k] += bik

    Vi    = V_pu[:, None]
    Vk    = V_pu[None, :]
    th_ik = th_rad[:, None] - th_rad[None, :]

    sin_ik = np.sin(th_ik)
    cos_ik = np.cos(th_ik)

    JPt =  Vi * Vk * ( g * sin_ik - b * cos_ik)
    JPU =  Vi *      ( g * cos_ik + b * sin_ik)
    JQt =  Vi * Vk * (-g * cos_ik - b * sin_ik)
    JQU =  Vi *      ( g * sin_ik - b * cos_ik)

    JPt_offdiag_sum = JPt.copy(); np.fill_diagonal(JPt_offdiag_sum, 0)
    JQU_offdiag_sum = JQU.copy(); np.fill_diagonal(JQU_offdiag_sum, 0)
    JPU_offdiag_sum = JPU.copy(); np.fill_diagonal(JPU_offdiag_sum, 0)
    JQt_offdiag_sum = JQt.copy(); np.fill_diagonal(JQt_offdiag_sum, 0)

    np.fill_diagonal(JPt, -JPt_offdiag_sum.sum(axis=1))
    np.fill_diagonal(JQU, -JQU_offdiag_sum.sum(axis=1) + 2.0 * V_pu * np.diag(b))
    np.fill_diagonal(JPU,  JPU_offdiag_sum.sum(axis=1) + V_pu * np.diag(g))
    np.fill_diagonal(JQt, -JQt_offdiag_sum.sum(axis=1))

    J = np.block([[JPt, JPU], [JQt, JQU]])
    return J, {"JPt": JPt, "JPU": JPU, "JQt": JQt, "JQU": JQU}


# ─────────────────────────────────────────────────────────────────────────────
# Grid class
# ─────────────────────────────────────────────────────────────────────────────

class WasterkingenGrid:
    """
    Wasterkingen distribution grid.
    Loads real LV data directly from the .xlsx file (no Pandapower).
    """

    def __init__(self, data_dir: Path = Path(__file__).parent.parent):
        self.data_dir = Path(data_dir)

        self.bus_df, self.line_df, self.load_df, self.ext_grid_df = self._load_dataframes()

        self.n_buses = len(self.bus_df)
        self.n_lines = len(self.line_df)
        self.pcc_bus = int(self.ext_grid_df['bus'].iloc[0])

        self.J        = None
        self.J_blocks = None

        self._BESS_DATA, self._PV_DATA, self._LOAD_DATA = self._make_device_data()

    # ── bus index helpers ─────────────────────────────────────────────────────

    def bus_pos(self, bus_idx: int) -> int:
        """Bus index → row/column position in the Jacobian matrix."""
        return self.bus_df.index.get_loc(bus_idx)

    @property
    def pcc_pos(self) -> int:
        """Positional index of the PCC (slack) bus in the Jacobian."""
        return self.bus_pos(self.pcc_bus)

    # ── network loading ───────────────────────────────────────────────────────

    def _load_dataframes(self):
        """Load bus, line, load, and ext_grid tables directly from Excel."""
        excel_path = self.data_dir / "70-1_0_4_grid.xlsx"

        if not excel_path.exists():
            raise FileNotFoundError(
                f"Grid data not found at '{excel_path.absolute()}'. "
                "Expected the 70-1_0_4_grid.xlsx file in the project root."
            )

        print(f"  Loaded real grid directly from {excel_path.name}")
        dfs = pd.read_excel(
            excel_path,
            sheet_name=['bus', 'line', 'load', 'ext_grid'],
            index_col=0,
            engine="openpyxl",
        )

        bus_df      = dfs['bus']
        line_df     = dfs['line']
        load_df     = dfs['load']
        ext_grid_df = dfs['ext_grid']

        if 'in_service' in line_df.columns:
            line_df = line_df[line_df['in_service'] == True].copy()

        return bus_df, line_df, load_df, ext_grid_df

    # ── device placement ──────────────────────────────────────────────────────

    def _make_device_data(self):
        """Derive device placement from the loaded LV network."""
        load_df_sorted = (self.load_df[self.load_df['bus'] != self.pcc_bus]
                          .copy()
                          .sort_values("p_mw", ascending=False)
                          .reset_index(drop=True))

        p_total = max(float(self.load_df['p_mw'].sum()), 0.01)

        # BESS: 40 % of total network load, 8-h energy capacity
        top_bus = int(load_df_sorted.iloc[0]['bus'])
        bess_s  = round(max(p_total * 0.40, 0.005), 5)
        bess = [{"bus": top_bus,
                 "S_max_MW":  bess_s,
                 "E_cap_MWh": bess_s * 8.0,
                 "soc_min": 0.1, "soc_max": 0.9, "soc_init": 0.5}]

        # PV: 50 % of total network load at top-4 load buses
        pv_s = round(max(p_total * 0.50, 0.003), 5)
        pv = [{"bus": int(load_df_sorted.iloc[i]['bus']),
               "S_max_MW": pv_s,
               "pf_min": 0.9}
              for i in range(min(4, len(load_df_sorted)))]

        # Controllable loads: 50 % of total network load, next 5 buses
        load_s = round(max(p_total * 0.50, 0.002), 5)
        load = [{"bus": int(load_df_sorted.iloc[i]['bus']),
                 "P_max_MW": load_s,
                 "pf": 0.95, "device_type": "hp"}
                for i in range(min(4, len(load_df_sorted)), min(9, len(load_df_sorted)))]

        return bess, pv, load

    # ── Jacobian ──────────────────────────────────────────────────────────────

    def build_jacobian(self):
        self.J, self.J_blocks = _build_jacobian(self.bus_df, self.line_df)
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

            for d in self._PV_DATA:
                b = d["bus"]
                row[f"bus_{b}_P_pv_base"] = d["S_max_MW"] * irr / 1000.0
                row[f"bus_{b}_Q_pv_base"] = 0.0

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
        bus_vn     = self.bus_df['vn_kv']
        from_buses = self.line_df['from_bus'].values
        v_kv = np.array([bus_vn.at[b] if b in bus_vn.index else 0.4
                         for b in from_buses], dtype=float)
        return self.line_df['max_i_ka'].values * v_kv * np.sqrt(3)

    @property
    def v_limits(self) -> tuple:
        """Voltage band for linearised constraints (widened to absorb flat-start error)."""
        return (0.85, 1.15)
