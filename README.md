# Impact of Grid Topology on the Flexibility of Wasterkingen's Electrical Grid

**Group:** Li Shan Ng, Yujin Kim, Lucas Laser  

## Overview

This project analyses how changing the grid topology affects the flexibility of Wasterkingen's LV distribution grid, measured via the Feasible Flexibility Operating Region (FFOR) at the Point of Common Coupling (PCC, bus 81).

Three scenarios are explored:
- **Scenario 1** – Baseline FFOR with varying device location and quantity
- **Scenario 2** – Optimal placement of a new distribution line
- **Scenario 3** – Effect of varying the new line's capacity at the optimal location

## Requirements

- Python 3.10+
- [Gurobi](https://www.gurobi.com/) with a valid licence
- Python packages:

```
pip install numpy pandas matplotlib scipy networkx openpyxl gurobipy
```

## Data

Grid data is loaded from `70-1_0_4_grid.xlsx` (Swiss-PDGs database, Grid ID: 70-1_0_4) located in the project root. The file must contain sheets: `bus`, `line`, `load`, `ext_grid`, `bus_geodata`.

## Running the Code

```bash
cd test
python main.py
```

The script will prompt:
```
Automatic placement? (y/n) [default: y]:
```
- Enter **`y`** for automatic placement (scattered, farthest leaves from PCC)
- Enter **`n`** for manual placement (enter bus numbers one by one)

### Case Configuration

Before each run, edit the two lines marked in `main.py`:

| Case | `n_bess` | `n_pv` | `n_loads` | Placement | `OUTPUT_DIR` |
|---|---|---|---|---|---|
| Figure 2 – single PV at bus 32/84/124 | 0 | 1 | 0 | n → enter bus | `results_1PV_fixed at 32` |
| Figure 2 – single HP at bus 32/84/124 | 0 | 0 | 1 | n → enter bus | `results_1HP_fixed at 32` |
| Figure 2 – single BESS at bus 32/84/124 | 1 | 0 | 0 | n → enter bus | `results_1BESS_fixed at 32` |
| Case 1 – 12PV, 1HP, 1BESS scattered | 1 | 12 | 1 | y | `results_12PV1HP1BESS` |
| Case 2 – 12PV, 1HP, 1BESS concentrated | 1 | 12 | 1 | n → enter southern buses | `results_12PV1HP1BESS_concentrated` |
| Case 3 – 27PV, 1HP, 1BESS scattered | 1 | 27 | 1 | y | `results_27PV1HP1BESS` |

## Output

Each run saves the following plots to `OUTPUT_DIR/`:

| File | Description |
|---|---|
| `ffor_single.png` | Single-timestep FFOR (Scenario 1) |
| `ffor_multi.png` | Multi-duration FFOR: 15 min, 30 min, 1 h, 2 h |
| `ffor_comparison.png` | FFOR before/after new line (Scenario 2) |
| `network_map.png` | Grid topology with device locations and new line |
| `ffor_capacity_overlay.png` | FFOR for different line capacities (Scenario 3) |
| `ffor_area_vs_capacity.png` | FFOR area vs. line capacity curve (Scenario 3) |

## Project Structure

```
.
├── 70-1_0_4_grid.xlsx          # Grid data (Swiss-PDGs)
├── test/
│   ├── main.py                 # Entry point
│   ├── grid_model.py           # Grid loading, Jacobian, device setup, synthetic baseline
│   ├── ffor_optimizer.py       # FFOR LP solver (single & multi-timestep)
│   ├── devices.py              # BESS, PV, HP device models and constraints
│   ├── line_add.py             # Optimal line placement and capacity sweep
│   ├── plot_network.py         # Network map visualisation
│   └── utils.py                # FFOR plotting helpers
└── results_*/                  # Output folders (one per case)
```

## Key Parameters

| Parameter | Value |
|---|---|
| Base power | 1 MVA |
| Base voltage | 0.4 kV |
| Voltage limits | 0.90 – 1.10 pu |
| FFOR directions | 36 |
| FFOR durations | 0.25 h, 0.5 h, 1 h, 2 h |
| PV capacity | 0.5 MW/unit |
| HP capacity | 0.2 MW/unit |
| BESS capacity | 0.1 MW / 0.4 MWh |
| Line constraint | 8-sided polygon inner approximation |
