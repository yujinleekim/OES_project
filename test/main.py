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
from line_add import find_optimal_line_addition, plot_ffor_comparison, compute_ffor_capacity_sweep, \
                    _add_candidate_line, _ffor_area
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
    grid._BESS_DATA, grid._PV_DATA, grid._LOAD_DATA = grid._make_device_data(n_bess=1, n_pv=12, n_loads=1)
    grid.build_jacobian()

    # 2. Load devices
    bess_list = grid.get_bess_devices()
    pv_list   = grid.get_pv_devices()
    load_list = grid.get_controllable_loads()
    baseline  = grid.get_baseline()

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
        from line_add import _median_line_params
        line_params = _median_line_params(grid.line_df)
        best_len    = line_result["best_length_km"]

        print("\n" + "=" * 50)
        print("  NEW LINE — Optimal Placement (Scenario 2)")
        print("=" * 50)
        print(f"  Route          : bus {best_bus_i} ↔ bus {best_bus_j}")
        print(f"  Length         : {best_len:.4f} km")
        print(f"  Capacity       : {line_params['max_i_ka']:.4f} kA  ({line_params['max_i_ka'] / 1.4434:.4f} pu)")
        print(f"  R              : {line_params['r_ohm_per_km'] * best_len:.4f} Ω  ({line_params['r_ohm_per_km']:.4f} Ω/km)")
        print(f"  X              : {line_params['x_ohm_per_km'] * best_len:.4f} Ω  ({line_params['x_ohm_per_km']:.4f} Ω/km)")
        print("-" * 50)
        print("  FFOR Results")
        print(f"  Base area      : {line_result['base_area']:.6f} pu²")
        print(f"  New area       : {line_result['best_area']:.6f} pu²")
        print(f"  Gain (abs)     : {line_result['area_gain']:+.6f} pu²")
        print(f"  Gain (%)       : {line_result['area_gain_pct']:+.2f}%")
        print("=" * 50)
        plot_network(
            grid=grid,
            new_line=line_result["best_pair"],
            save_path=OUTPUT_DIR / "network_map.png",
        )

        # Capacity sweep at fixed best location
        print("\n[6/6] Capacity sweep at optimal line location ...")
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

        # Multi-duration FFOR comparison: Scenario 1 vs Scenario 2
        print("\n  Computing multi-duration FFOR for Scenario 2 ...")
        line_df_orig = grid.line_df.copy()
        line_df_with = line_df_orig.copy()
        _add_candidate_line(line_df_with, best_bus_i, best_bus_j, best_len, line_params)
        grid.line_df, grid.n_lines = line_df_with, len(line_df_with)
        grid.build_jacobian()

        opt_s2 = FFOROptimizer(grid, bess_list, pv_list, load_list)
        multi_s2 = {d: opt_s2.compute_ffor_multi_timestep(t_start=t_idx, duration_h=d, baseline=baseline, n_directions=36)
                    for d in durations_h}

        grid.line_df, grid.n_lines = line_df_orig, len(line_df_orig)
        grid.build_jacobian()

        print("\n" + "=" * 65)
        print("  FFOR by Duration: Scenario 1 (base) vs Scenario 2 (new line)")
        print("=" * 65)
        print(f"  {'Duration':<10} {'Sc.1 Area (pu²)':<18} {'Sc.2 Area (pu²)':<18} {'Gain (%)'}")
        print("-" * 65)
        for d in durations_h:
            label = f"{d}h" if d >= 1 else f"{int(d*60)}min"
            a1 = _ffor_area(multi_results[d].P_flex, multi_results[d].Q_flex)
            a2 = _ffor_area(multi_s2[d].P_flex, multi_s2[d].Q_flex)
            gain_pct = 100.0 * (a2 - a1) / a1 if a1 > 0 else float("inf")
            print(f"  {label:<10} {a1:<18.6f} {a2:<18.6f} {gain_pct:+.2f}%")
        print("=" * 65)

    print(f"\nDone. All figures saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()