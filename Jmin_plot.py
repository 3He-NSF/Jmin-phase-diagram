import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Matplotlib settings
# ============================================================
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})


# ============================================================
# Scan ranges
# ============================================================
J2_OVER_J1_MIN = -2.0
J2_OVER_J1_MAX = 2.0
N_J2 = 201

J3_OVER_J1_MIN = -2.0
J3_OVER_J1_MAX = 2.0
N_J3 = 201

Q_MIN = -0.5
Q_MAX = 0.5
Q_STEP = 0.02


# ============================================================
# Model
# ============================================================
def j_minus(H, K, L, J2_over_J1, J3_over_J1):
    """Return the AFM-like branch J_-(q) with J1 fixed to 1."""
    return (
        -16.0 * np.cos(np.pi * H) * np.cos(np.pi * K) * np.cos(np.pi * L)
        + 4.0 * J2_over_J1 * np.cos(2.0 * np.pi * L)
        + 4.0
        * J3_over_J1
        * (np.cos(2.0 * np.pi * H) + np.cos(2.0 * np.pi * K))
    )


def calculate_minimum_maps(J2_values, J3_values, q_values):
    """
    Scan q space and find q_min for each point in the J2/J1-J3/J1 parameter space.

    The returned maps have the shape (len(J3_values), len(J2_values)).
    This shape is convenient for imshow with x = J2/J1 and y = J3/J1.
    """
    J2_grid, J3_grid = np.meshgrid(J2_values, J3_values, indexing="xy")

    energy_min_map = np.full((len(J3_values), len(J2_values)), np.inf)
    H_min_map = np.zeros_like(energy_min_map)
    K_min_map = np.zeros_like(energy_min_map)
    L_min_map = np.zeros_like(energy_min_map)

    for H in q_values:
        cos_pi_H = np.cos(np.pi * H)
        cos_2pi_H = np.cos(2.0 * np.pi * H)

        for K in q_values:
            cos_pi_K = np.cos(np.pi * K)
            cos_2pi_K = np.cos(2.0 * np.pi * K)

            for L in q_values:
                base_energy = -16.0 * cos_pi_H * cos_pi_K * np.cos(np.pi * L)
                J2_coefficient = 4.0 * np.cos(2.0 * np.pi * L)
                J3_coefficient = 4.0 * (cos_2pi_H + cos_2pi_K)

                # J2_grid and J3_grid are 2D arrays with shape (N_J3, N_J2).
                # This calculates J(q) for all parameter points at the current q.
                energy_values = (
                    base_energy
                    + J2_coefficient * J2_grid
                    + J3_coefficient * J3_grid
                )

                update_mask = energy_values < energy_min_map

                energy_min_map[update_mask] = energy_values[update_mask]
                H_min_map[update_mask] = round(H, 2)
                K_min_map[update_mask] = round(K, 2)
                L_min_map[update_mask] = round(L, 2)

    return energy_min_map, H_min_map, K_min_map, L_min_map


def make_q_min_rgb_map(H_min_map, K_min_map, L_min_map, map_correction="True"):
    """
    Convert q_min = (H, K, L) into an RGB image.

    The absolute value is used so that +q and -q have the same color.
    Each component is mapped as
        red   <- |H_min|
        green <- |K_min|
        blue  <- |L_min|
    """
    base_brightness = 0.4
    contrast = 1.5

    rgb_map = np.zeros((*H_min_map.shape, 3), dtype=float)
    if map_correction == "True":
        rgb_map[..., 0] = base_brightness + contrast * np.abs(H_min_map)
        rgb_map[..., 1] = base_brightness + contrast * np.abs(K_min_map)
        rgb_map[..., 2] = base_brightness + contrast * np.abs(L_min_map)
    else:
        rgb_map[..., 0] = np.abs(H_min_map)
        rgb_map[..., 1] = np.abs(K_min_map)
        rgb_map[..., 2] = np.abs(L_min_map)

    return rgb_map


def plot_q_min_map(rgb_map, output_filename):
    """Plot and save the RGB map of q_min."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.imshow(
        rgb_map,
        origin="lower",
        extent=[J2_OVER_J1_MIN, J2_OVER_J1_MAX, J3_OVER_J1_MIN, J3_OVER_J1_MAX],
        aspect="auto",
    )

    ax.set_xlabel(r"$J_2/J_1$")
    ax.set_ylabel(r"$J_3/J_1$")
    ax.set_title(r"$q_{\min}$ map")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_filename, dpi=300)


def plot_energy_min_map(energy_min_map, output_filename):
    """Plot and save the minimum-energy map J(q_min)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    image = ax.imshow(
        energy_min_map,
        origin="lower",
        extent=[J2_OVER_J1_MIN, J2_OVER_J1_MAX, J3_OVER_J1_MIN, J3_OVER_J1_MAX],
        aspect="auto",
    )

    fig.colorbar(image, ax=ax, label=r"$J_{\min}$")
    ax.set_xlabel(r"$J_2/J_1$")
    ax.set_ylabel(r"$J_3/J_1$")
    ax.set_title(r"$J(q_{\min})$")

    fig.tight_layout()
    fig.savefig(output_filename, dpi=300)


# ============================================================
# Main calculation
# ============================================================
J2_over_J1_values = np.linspace(J2_OVER_J1_MIN, J2_OVER_J1_MAX, N_J2)
J3_over_J1_values = np.linspace(J3_OVER_J1_MIN, J3_OVER_J1_MAX, N_J3)
q_values = np.arange(Q_MIN, Q_MAX + 0.5 * Q_STEP, Q_STEP, dtype=float)

E_min_map, H_min_map, K_min_map, L_min_map = calculate_minimum_maps(
    J2_over_J1_values,
    J3_over_J1_values,
    q_values,
)

rgb_map = make_q_min_rgb_map(H_min_map, K_min_map, L_min_map, map_correction="True")

plot_q_min_map(rgb_map, "q_min_rgb_map.pdf")
plot_energy_min_map(E_min_map, "J_min_map.pdf")

plt.show()