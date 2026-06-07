"""
utils.py - Plotting helper functions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

def _close_polygon(P, Q):
    if len(P) == 0: return P, Q
    return np.append(P, P[0]), np.append(Q, Q[0])

def _convex_hull_order(P, Q):
    """Sort points by polar angle around centroid for correct polygon rendering."""
    mask = ~(np.isnan(P) | np.isnan(Q))
    P_clean = P[mask]
    Q_clean = Q[mask]

    if len(P_clean) < 3:
        return P_clean, Q_clean

    # Compute centroid
    p_center = np.mean(P_clean)
    q_center = np.mean(Q_clean)

    # Compute polar angles relative to centroid
    angles = np.arctan2(Q_clean - q_center, P_clean - p_center)

    # Sort points by angle
    sort_idx = np.argsort(angles)
    return P_clean[sort_idx], Q_clean[sort_idx]

def plot_ffor(result, title: str = "FFOR", save_path: Path = None):
    if len(result.P_flex) == 0:
        print(f"  [INFO] FFOR result is empty — skipping plot '{title}'")
        return
        
    P, Q = _convex_hull_order(result.P_flex, result.Q_flex)
    P, Q = _close_polygon(P, Q)

    fig, ax = plt.subplots(figsize=(7, 7))
    
    ax.fill(P, Q, alpha=0.3, color="steelblue", label="FFOR Area")
    ax.plot(P, Q, "b-", linewidth=2.0)

    margin = 0.1
    ax.set_xlim(min(P) - margin, max(P) + margin)
    ax.set_ylim(min(Q) - margin, max(Q) + margin)
    
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax.plot(0, 0, "ro", markersize=8, label="Baseline (0,0)")
    
    ax.set_xlabel("Active Power Flexibility [pu]")
    ax.set_ylabel("Reactive Power Flexibility [pu]")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path, dpi=150)
        print(f"  [PLOT] Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_ffor_multi_duration(ffor_multi: dict, title: str = "Multi-Timestep FFOR", save_path: Path = None):
    durations = sorted(ffor_multi.keys())
    cmap = cm.get_cmap("plasma", len(durations) + 1)

    fig, ax = plt.subplots(figsize=(7, 7))

    for i, d in enumerate(durations):
        res = ffor_multi[d]
        if len(res.P_flex) < 3:
            continue
            
        P, Q = _convex_hull_order(res.P_flex, res.Q_flex)
        P, Q = _close_polygon(P, Q)
        
        label = f"{d} h" if d >= 1 else f"{int(d*60)} min"
        color = cmap(i)
        
        ax.fill(P, Q, alpha=0.15, color=color)
        ax.plot(P, Q, color=color, linewidth=1.8, label=label)

    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.plot(0, 0, "ko", markersize=6)
    
    ax.set_xlabel("Active Power Flexibility [pu]")
    ax.set_ylabel("Reactive Power Flexibility [pu]")
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(loc="upper right", title="Sustained Duration")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_active_flex_vs_duration(ffor_multi: dict, save_path: Path = None):
    durations = sorted(ffor_multi.keys())
    p_pos = [max(ffor_multi[d].P_flex, default=0) for d in durations]
    p_neg = [min(ffor_multi[d].P_flex, default=0) for d in durations]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(durations, p_pos, "b-o", linewidth=1.5, label="Positive Flex (+P)")
    ax.plot(durations, p_neg, "r-o", linewidth=1.5, label="Negative Flex (-P)")
    ax.set_xlabel("Sustained Duration [h]")
    ax.set_ylabel("Active Power Flexibility [pu]")
    ax.set_title("Flexibility Parameter vs. Duration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_ffor_multi_duration(ffor_results, save_path=None):
    """ffor_results: dict with key=duration (e.g. 0.5, 1, 2, 4) and value=FFORResult"""
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(ffor_results)))

    for i, (duration, res) in enumerate(ffor_results.items()):
        P, Q = _convex_hull_order(res.P_flex, res.Q_flex)
        P, Q = _close_polygon(P, Q)
        ax.plot(P, Q, color=colors[i], linewidth=2.0, label=f"Duration: {duration}h")

    ax.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Active Power Flexibility [pu]")
    ax.set_ylabel("Reactive Power Flexibility [pu]")
    ax.set_title("Effect of Sustained Duration on FFOR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path: fig.savefig(save_path, dpi=200)
    plt.close(fig)

def plot_ffor_capacity_overlay(capacity_results: dict, base_ffor=None,
                               bus_pair: tuple = None, save_path=None):
    """FFOR 오버레이: 용량별 폴리곤을 한 플롯에 겹쳐서 표시."""
    fig, ax = plt.subplots(figsize=(7, 7))
    caps    = sorted(capacity_results.keys())
    cmap    = cm.get_cmap("plasma", len(caps) + 2)

    if base_ffor is not None and len(base_ffor.P_flex) >= 3:
        P, Q = _close_polygon(*_convex_hull_order(base_ffor.P_flex, base_ffor.Q_flex))
        ax.fill(P, Q, alpha=0.15, color="steelblue")
        ax.plot(P, Q, color="steelblue", linewidth=2.0, linestyle="--", label="Baseline (no new line)")

    for i, cap in enumerate(caps):
        res = capacity_results[cap]
        if len(res.P_flex) < 3:
            continue
        P, Q = _close_polygon(*_convex_hull_order(res.P_flex, res.Q_flex))
        color = cmap(i + 1)
        ax.fill(P, Q, alpha=0.12, color=color)
        ax.plot(P, Q, color=color, linewidth=1.8, label=f"{cap:.3f} kA")

    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax.plot(0, 0, "ko", markersize=5)
    ax.set_xlabel("Active Power Flexibility [pu]")
    ax.set_ylabel("Reactive Power Flexibility [pu]")
    title = f"FFOR by Line Capacity – bus {bus_pair[0]}↔{bus_pair[1]}" if bus_pair else "FFOR by Line Capacity"
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", title="max_i_ka", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_ffor_area_vs_capacity(capacity_results: dict, base_area: float = None,
                                bus_pair: tuple = None, save_path=None):
    """FFOR 면적 vs 라인 용량 그래프."""
    from scipy.spatial import ConvexHull

    def _area(res):
        mask = ~(np.isnan(res.P_flex) | np.isnan(res.Q_flex))
        pts  = np.column_stack([res.P_flex[mask], res.Q_flex[mask]])
        if len(pts) < 3:
            return 0.0
        try:
            return ConvexHull(pts).volume
        except Exception:
            return 0.0

    caps  = sorted(capacity_results.keys())
    areas = [_area(capacity_results[c]) for c in caps]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(caps, areas, "o-", color="tomato", linewidth=2, markersize=7, label="With new line")

    if base_area is not None:
        ax.axhline(base_area, color="steelblue", linewidth=1.5,
                   linestyle="--", label=f"Baseline ({base_area:.4f})")

    ax.set_xlabel("New Line Capacity [kA]")
    ax.set_ylabel("FFOR Area [pu²]")
    title = f"FFOR Area vs. Line Capacity – bus {bus_pair[0]}↔{bus_pair[1]}" if bus_pair else "FFOR Area vs. Line Capacity"
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_ffor_comparison_full(base_ffor, opt_ffor, label_opt="With new line", save_path=None):
    fig, ax = plt.subplots(figsize=(7, 7))

    P_b, Q_b = _close_polygon(*_convex_hull_order(base_ffor.P_flex, base_ffor.Q_flex))
    ax.fill(P_b, Q_b, color="steelblue", alpha=0.2, label="Baseline FFOR")
    ax.plot(P_b, Q_b, color="steelblue", linewidth=2)

    P_o, Q_o = _close_polygon(*_convex_hull_order(opt_ffor.P_flex, opt_ffor.Q_flex))
    ax.fill(P_o, Q_o, color="tomato", alpha=0.2, label=label_opt)
    ax.plot(P_o, Q_o, color="tomato", linewidth=2)

    ax.set_title("FFOR Before/After Comparison")
    ax.set_xlabel("P_flex [pu]")
    ax.set_ylabel("Q_flex [pu]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path: fig.savefig(save_path, dpi=200)
    plt.close(fig)