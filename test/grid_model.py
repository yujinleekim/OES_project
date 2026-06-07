"""
grid_model.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from devices import BESSModel, PVModel, ControllableLoadModel

def _build_jacobian(bus_df: pd.DataFrame, line_df: pd.DataFrame):
    n = len(bus_df)
    V_pu, th_rad = np.ones(n), np.zeros(n)
    g, b = np.zeros((n, n)), np.zeros((n, n))
    bus_idx = {idx: pos for pos, idx in enumerate(bus_df.index)}

    for _, row in line_df.iterrows():
        if 'in_service' in row and not row['in_service']: continue
        i, k = bus_idx[int(row['from_bus'])], bus_idx[int(row['to_bus'])]
        Z_base_ohm = (0.4 ** 2) / 1.0
        
        length = max(float(row.get('length_km', 0.001)), 0.001)
        r_ohm = max(float(row.get('r_ohm_per_km', 0.001)), 0.001) * length
        x_ohm = max(float(row.get('x_ohm_per_km', 0.001)), 0.001) * length
        
        z = complex(r_ohm, x_ohm) / Z_base_ohm
        y = 1.0 / z
        gik, bik = y.real, y.imag
        g[i, k] -= gik;  g[k, i] -= gik
        g[i, i] += gik;  g[k, k] += gik
        b[i, k] -= bik;  b[k, i] -= bik
        b[i, i] += bik;  b[k, k] += bik

    JPt, JPU = -b.copy(), g.copy()
    JQt, JQU = -g.copy(), -b.copy()

    J = np.block([[JPt, JPU], [JQt, JQU]])
    return J, {"JPt": JPt, "JPU": JPU, "JQt": JQt, "JQU": JQU}

class WasterkingenGrid:
    def __init__(self, data_dir: Path = Path(__file__).parent.parent):
        self.data_dir = Path(data_dir)
        self.bus_df, self.line_df, self.load_df, self.ext_grid_df = self._load_dataframes()
        self.pcc_bus = int(self.ext_grid_df['bus'].iloc[0])

        import networkx as nx
        G = nx.Graph()
        for _, r in self.line_df.iterrows():
            G.add_edge(int(r['from_bus']), int(r['to_bus']))
        
        if self.pcc_bus in G:
            connected = set(nx.node_connected_component(G, self.pcc_bus))
        else:
            connected = max(list(nx.connected_components(G)), key=len)
            self.pcc_bus = list(connected)[0]
            
        self.bus_df = self.bus_df[self.bus_df.index.isin(connected)].copy()
        self.line_df = self.line_df[self.line_df['from_bus'].isin(connected) & self.line_df['to_bus'].isin(connected)].copy()
        self.load_df = self.load_df[self.load_df['bus'].isin(connected)].copy()

        self.n_buses = len(self.bus_df)
        self.n_lines = len(self.line_df)
        self.J, self.J_blocks = None, None
        self._BESS_DATA, self._PV_DATA, self._LOAD_DATA = [], [], []

    def bus_pos(self, bus_idx: int) -> int: return self.bus_df.index.get_loc(bus_idx)
    @property
    def pcc_pos(self) -> int: return self.bus_pos(self.pcc_bus)

    def _load_dataframes(self):
        excel_path = self.data_dir / "70-1_0_4_grid.xlsx"
        dfs = pd.read_excel(excel_path, sheet_name=['bus', 'line', 'load', 'ext_grid'], index_col=0, engine="openpyxl")
        bus_df, line_df, load_df, ext_grid_df = dfs['bus'], dfs['line'], dfs['load'], dfs['ext_grid']
        if 'in_service' in line_df.columns: line_df = line_df[line_df['in_service'] == True].copy()
        return bus_df, line_df, load_df, ext_grid_df

    def _make_device_data(self, n_bess=2, n_pv=10, n_loads=20):
        import networkx as nx
        G = nx.Graph()
        for _, r in self.line_df.iterrows(): 
            G.add_edge(int(r['from_bus']), int(r['to_bus']))
        
        leaf_nodes = [node for node, degree in G.degree() if degree == 1 and node != self.pcc_bus]
        if not leaf_nodes: 
            leaf_nodes = [n for n in G.nodes() if n != self.pcc_bus] or [self.pcc_bus]

        # Sort leaves by distance from PCC (farthest first)
        leaf_nodes = sorted(leaf_nodes, key=lambda x: nx.shortest_path_length(G, x, self.pcc_bus), reverse=True)
        print(f"\n--- Device Configuration ---")
        mode = input("Automatic placement? (y/n) [default: y]: ").lower().strip() or 'y'

        bess, pv, load = [], [], []

        if mode == 'y':
            # Automatic placement
            def get_bus(idx, offset=0):
                return int(leaf_nodes[(idx + offset) % len(leaf_nodes)])

            bess_s = 0.1
            bess = [{"bus": get_bus(i), "S_max_MW": bess_s, "E_cap_MWh": bess_s * 4.0,
                     "soc_min": 0.1, "soc_max": 0.9, "soc_init": 0.5} for i in range(n_bess)]
            pv = [{"bus": get_bus(i, offset=n_bess), "S_max_MW": 0.5, "pf_min": 0.9} for i in range(n_pv)]
            load = [{"bus": get_bus(i, offset=n_bess + n_pv), "P_max_MW": 0.2, "pf": 0.95, "device_type": "hp"} for i in range(n_loads)]
        else:
            # Manual input
            print(f"Available buses: {sorted(list(G.nodes()))}")

            for i in range(n_bess):
                bus = int(input(f"Bus for BESS {i+1}: "))
                bess.append({"bus": bus, "S_max_MW": 0.1, "E_cap_MWh": 0.4, "soc_min": 0.1, "soc_max": 0.9, "soc_init": 0.5})

            for i in range(n_pv):
                bus = int(input(f"Bus for PV {i+1}: "))
                pv.append({"bus": bus, "S_max_MW": 0.5, "pf_min": 0.9})

            for i in range(n_loads):
                bus = int(input(f"Bus for Load {i+1}: "))
                load.append({"bus": bus, "P_max_MW": 0.2, "pf": 0.95, "device_type": "hp"})

        return bess, pv, load
    
    def build_jacobian(self):
        self.J, self.J_blocks = _build_jacobian(self.bus_df, self.line_df)
        return self.J

    def get_bess_devices(self): return [BESSModel(**d) for d in self._BESS_DATA]
    def get_pv_devices(self): return [PVModel(**d) for d in self._PV_DATA]
    def get_controllable_loads(self): return [ControllableLoadModel(**d) for d in self._LOAD_DATA]

    def get_baseline(self, date: str = "2021-09-01", resolution_min: int = 15) -> pd.DataFrame:
        csv_file = self.data_dir / f"baseline_{date}.csv"
        if csv_file.exists(): return pd.read_csv(csv_file, parse_dates=["time"])
        return self._synthetic_baseline(date, resolution_min)

    def _synthetic_baseline(self, date: str, resolution_min: int) -> pd.DataFrame:
        n_steps = 24 * 60 // resolution_min
        t = np.linspace(0, 24, n_steps, endpoint=False)
        irradiance = np.clip(800 * np.exp(-0.5 * ((t - 12) / 3) ** 2), 0, None)
        temperature = 15 + 8 * np.sin(2 * np.pi * (t - 6) / 24)

        rows = []
        for ti, irr, temp in zip(t, irradiance, temperature):
            row = {"time": ti, "irradiance": irr, "temperature": temp}
            for d in self._PV_DATA:
                b = d["bus"]
                row[f"bus_{b}_P_pv_base"] = d["S_max_MW"] * irr / 1000.0
                row[f"bus_{b}_Q_pv_base"] = 0.0
            for d in self._LOAD_DATA:
                b = d["bus"]
                tan_phi = np.tan(np.arccos(d["pf"]))
                p_hp = d["P_max_MW"] * max(0, (15 - temp) / 23) if d["device_type"] == "hp" else d["P_max_MW"] * 0.3
                row[f"bus_{b}_P_hp_base"] = p_hp
                row[f"bus_{b}_Q_hp_base"] = p_hp * tan_phi
            rows.append(row)
        return pd.DataFrame(rows)

    @property
    def line_ratings_mva(self) -> np.ndarray:
        bus_vn = self.bus_df['vn_kv']
        from_buses = self.line_df['from_bus'].values
        v_kv = np.array([bus_vn.at[b] if b in bus_vn.index else 0.4 for b in from_buses], dtype=float)
        max_i = self.line_df['max_i_ka'].fillna(0.2).values
        ratings = max_i * v_kv * np.sqrt(3)
        lengths = self.line_df['length_km'].values
        ratings[lengths <= 0.05] = 999.0
        return ratings

    @property
    def v_limits(self) -> tuple: return (0.90, 1.10)