"""
FFOR Computation – main.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Eigene Module
from grid_model import WasterkingenGrid
from ffor_optimizer import FFOROptimizer
from utils import plot_ffor, plot_ffor_multi_duration
from line_add import find_optimal_line_addition, plot_ffor_comparison

DATA_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print("=" * 60)
    print("FFOR Computation – Wasterkingen Distribution Grid")
    print("=" * 60)
    
    # 1. Grid laden
    grid = WasterkingenGrid(data_dir=DATA_DIR)
    grid._BESS_DATA, grid._PV_DATA, grid._LOAD_DATA = grid._make_device_data(n_bess=1, n_pv=12, n_loads=5)
    grid.build_jacobian()
    
    # 2. Geräte laden
    bess_list = grid.get_bess_devices()
    pv_list   = grid.get_pv_devices()
    load_list = grid.get_controllable_loads()
    baseline  = grid.get_baseline(date="2021-09-01", resolution_min=15)

    # 3. FFOR für den Ist-Zustand berechnen
    optimizer = FFOROptimizer(grid, bess_list, pv_list, load_list)
    t_idx = 56
    ffor_single = optimizer.compute_ffor_single_timestep(t_idx=t_idx, baseline=baseline, n_directions=36)

    # 4. Optimierung: Beste zusätzliche Leitung finden (MUSS VOR PLOTTING SEIN)
    print("\n[5/5] Searching for optimal line addition ...")
    line_result = find_optimal_line_addition(
        grid=grid, bess_list=bess_list, pv_list=pv_list, load_list=load_list, 
        baseline=baseline, t_idx=t_idx, n_directions=36, length_km=0.5, top_n=20, verbose=True
    )
    
    # 5. Plotting-Teil
    print("\nGenerating plots ...")
    
    # Single Plot
    if len(ffor_single.P_flex) > 3:
        plot_ffor(ffor_single, title="Single-Timestep FFOR", save_path=OUTPUT_DIR / "ffor_single.png")

    # Multi-Duration Plot
    durations_h = [0.25, 0.5, 1, 2]
    multi_results = {d: optimizer.compute_ffor_multi_timestep(t_start=t_idx, duration_h=d, baseline=baseline, n_directions=36) 
                     for d in durations_h}
    plot_ffor_multi_duration(multi_results, save_path=OUTPUT_DIR / "ffor_multi.png")
    
    # Vorher-Nachher Vergleich
    if line_result and "best_ffor" in line_result:
        # Hier muss der Funktionsname mit dem Import übereinstimmen!
        plot_ffor_comparison(
            base_ffor=ffor_single, 
            opt_ffor=line_result["best_ffor"], 
            best_pair=line_result["best_pair"],
            save_path=OUTPUT_DIR / "ffor_comparison.png"
        )
        print("[DEBUG] Comparison plot saved.")
        # Ausgabe des Ergebnisses
        best_bus_i, best_bus_j = line_result["best_pair"]
        print(f"\n  → Best new line: bus {best_bus_i} ↔ bus {best_bus_j}")
        print(f"    FFOR area gain: {line_result['area_gain']:+.6f} MW·MVAr")

    print(f"\nDone. All figures saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()