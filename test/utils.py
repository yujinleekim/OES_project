"""
utils.py - Bereinigte und robuste Plotting-Hilfsfunktionen
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

def _close_polygon(P, Q):
    if len(P) == 0: return P, Q
    return np.append(P, P[0]), np.append(Q, Q[0])

def _convex_hull_order(P, Q):
    """
    Sortiert die Punkte im Uhrzeigersinn nach ihrem Polarwinkel um den Schwerpunkt.
    Das ist extrem robust und verhindert kollabierte, weiße Plots bei flachen Polygonen.
    """
    mask = ~(np.isnan(P) | np.isnan(Q))
    P_clean = P[mask]
    Q_clean = Q[mask]
    
    if len(P_clean) < 3:
        return P_clean, Q_clean
        
    # Schwerpunkt (Center of Mass) berechnen
    p_center = np.mean(P_clean)
    q_center = np.mean(Q_clean)
    
    # Polarwinkel aller Punkte relativ zum Schwerpunkt berechnen
    angles = np.arctan2(Q_clean - q_center, P_clean - p_center)
    
    # Punkte nach Winkel sortieren
    sort_idx = np.argsort(angles)
    return P_clean[sort_idx], Q_clean[sort_idx]

def plot_ffor(result, title: str = "FFOR", save_path: Path = None):
    if len(result.P_flex) == 0:
        print(f"  [INFO] FFOR result is empty — skipping plot '{title}'")
        return
        
    P, Q = _convex_hull_order(result.P_flex, result.Q_flex)
    P, Q = _close_polygon(P, Q)

    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Hier zeichnen wir die Fläche
    ax.fill(P, Q, alpha=0.3, color="steelblue", label="FFOR Area")
    ax.plot(P, Q, "b-", linewidth=2.0)
    
    # WICHTIG: Manuelles Setzen der Limits basierend auf den Daten
    margin = 0.1 # 10% Rand
    ax.set_xlim(min(P) - margin, max(P) + margin)
    ax.set_ylim(min(Q) - margin, max(Q) + margin)
    
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax.plot(0, 0, "ro", markersize=8, label="Baseline (0,0)")
    
    ax.set_xlabel("Active Power Flexibility [MW]")
    ax.set_ylabel("Reactive Power Flexibility [MVAr]")
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
    
    ax.set_xlabel("Active Power Flexibility [MW]")
    ax.set_ylabel("Reactive Power Flexibility [MVAr]")
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
    ax.set_ylabel("Active Power Flexibility [MW]")
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
    """
    ffor_results: Dict mit Key=Dauer (z.B. 0.5, 1, 2, 4) und Value=FFOR-Objekt
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(ffor_results)))
    
    for i, (duration, res) in enumerate(ffor_results.items()):
        P, Q = _convex_hull_order(res.P_flex, res.Q_flex)
        P, Q = _close_polygon(P, Q)
        ax.plot(P, Q, color=colors[i], linewidth=2.0, label=f"Dauer: {duration}h")
    
    ax.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Active Power Flexibility [MW]")
    ax.set_ylabel("Reactive Power Flexibility [MVAr]")
    ax.set_title("Einfluss der Entladedauer auf die FFOR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path: fig.savefig(save_path, dpi=200)
    plt.close(fig)

def plot_ffor_comparison_full(base_ffor, opt_ffor, label_opt="Mit Netzausbau", save_path=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Baseline (Blau)
    P_b, Q_b = _close_polygon(*_convex_hull_order(base_ffor.P_flex, base_ffor.Q_flex))
    ax.fill(P_b, Q_b, color="steelblue", alpha=0.2, label="Baseline FFOR")
    ax.plot(P_b, Q_b, color="steelblue", linewidth=2)
    
    # Opt (Rot)
    P_o, Q_o = _close_polygon(*_convex_hull_order(opt_ffor.P_flex, opt_ffor.Q_flex))
    ax.fill(P_o, Q_o, color="tomato", alpha=0.2, label=label_opt)
    ax.plot(P_o, Q_o, color="tomato", linewidth=2)
    
    ax.set_title("FFOR Vorher-Nachher Vergleich")
    ax.set_xlabel("P_flex [MW]")
    ax.set_ylabel("Q_flex [MVAr]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path: fig.savefig(save_path, dpi=200)
    plt.close(fig)