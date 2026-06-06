"""
FFOR Computation – main.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Local modules
from grid_model import WasterkingenGrid
from ffor_optimizer import FFOROptimizer
from utils import plot_ffor, plot_ffor_multi_duration
from line_add import find_optimal_line_addition, plot_ffor_comparison, compute_ffor_capacity_sweep
from plot_network import plot_network
from utils import plot_ffor_capacity_overlay, plot_ffor_area_vs_capacity

DATA_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print("=" * 60)
    print("FFOR Computation – Wasterkingen Distribution Grid")
    print("=" * 60)
    
    # 1. Load grid
    grid = WasterkingenGrid(data_dir=DATA_DIR)
    grid._BESS_DATA, grid._PV_DATA, grid._LOAD_DATA = grid._make_device_data(n_bess=1, n_pv=12, n_loads=5)
    grid.build_jacobian()

    # 2. Load devices
    bess_list = grid.get_bess_devices()
    pv_list   = grid.get_pv_devices()
    load_list = grid.get_controllable_loads()
    baseline  = grid.get_baseline(date="2021-09-01", resolution_min=15)

    # 3. Compute baseline FFOR
    optimizer = FFOROptimizer(grid, bess_list, pv_list, load_list)
    t_idx = 56
    ffor_single = optimizer.compute_ffor_single_timestep(t_idx=t_idx, baseline=baseline, n_directions=36)

    # 4. Find optimal new line
    print("\n[5/5] Searching for optimal line addition ...")
    line_result = find_optimal_line_addition(
        grid=grid, bess_list=bess_list, pv_list=pv_list, load_list=load_list,
        baseline=baseline, t_idx=t_idx, n_directions=36, length_km=0.5, top_n=20, verbose=True
    )

    # 5. Plotting
    print("\nGenerating plots ...")

    # Single-timestep FFOR
    if len(ffor_single.P_flex) > 3:
        plot_ffor(ffor_single, title="Single-Timestep FFOR", save_path=OUTPUT_DIR / "ffor_single.png")

    # Multi-duration FFOR
    durations_h = [0.25, 0.5, 1, 2]
    multi_results = {d: optimizer.compute_ffor_multi_timestep(t_start=t_idx, duration_h=d, baseline=baseline, n_directions=36)
                     for d in durations_h}
    plot_ffor_multi_duration(multi_results, save_path=OUTPUT_DIR / "ffor_multi.png")

    # Before/after comparison
    if line_result and "best_ffor" in line_result:
        plot_ffor_comparison(
            base_ffor=ffor_single,
            opt_ffor=line_result["best_ffor"],
            best_pair=line_result["best_pair"],
            save_path=OUTPUT_DIR / "ffor_comparison.png"
        )
        # Print results
        best_bus_i, best_bus_j = line_result["best_pair"]
        print(f"\n  → Best new line: bus {best_bus_i} ↔ bus {best_bus_j}")
        print(f"    FFOR area gain: {line_result['area_gain']:+.6f} MW·MVAr")
        plot_network(
            grid=grid,
            new_line=line_result["best_pair"],
            save_path=OUTPUT_DIR / "network_map.png",
        )

        # Capacity sweep at fixed best location
        print("\n[6/6] Capacity sweep at optimal line location ...")
        from line_add import _median_line_params
        median_cap = _median_line_params(grid.line_df)["max_i_ka"]
        capacities = [median_cap * f for f in [0.25, 0.5, 1.0, 2.0, 4.0]]

        cap_results = compute_ffor_capacity_sweep(
            grid=grid, bess_list=bess_list, pv_list=pv_list, load_list=load_list,
            baseline=baseline, bus_i=best_bus_i, bus_j=best_bus_j,
            capacities_ka=capacities, length_km=line_result["best_length_km"], t_idx=t_idx, n_directions=36,
        )

        plot_ffor_capacity_overlay(
            cap_results, base_ffor=ffor_single, bus_pair=line_result["best_pair"],
            save_path=OUTPUT_DIR / "ffor_capacity_overlay.png",
        )
        plot_ffor_area_vs_capacity(
            cap_results, base_area=line_result["base_area"],
            bus_pair=line_result["best_pair"],
            save_path=OUTPUT_DIR / "ffor_area_vs_capacity.png",
        )

    print(f"\nDone. All figures saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()