#!/usr/bin/env python3
"""Example 1: Toggle Switch

Computes Morse graphs, Morse sets, and regions of attraction
for the toggle switch using different methods.

Methods:
    1. F_integration (f): Numerical ODE integration of the ground-truth system
    2. F_data: Data-driven approach with "outside" mapping for unmapped boxes
    3. F_Lipschitz (f): Lipschitz-based bounds using ground-truth system
    4. F_gaussianprocess (GP): GP-based outer approximation with confidence bounds (arXiv:2210.01292v1)
    5. F_integration (hat_f): Numerical integration of learned system from data
"""

import numpy as np
import os
import argparse
import time

from MorseGraphL4DC.grids import UniformGrid
from MorseGraphL4DC.dynamics import F_integration, F_Lipschitz, F_data, F_gaussianprocess
from MorseGraphL4DC.systems import SwitchingSystem
from MorseGraphL4DC.learning import train_gp_from_data
from MorseGraphL4DC.utils import generate_trajectory_data, define_tau_map, compute_modes
from MorseGraphL4DC.postprocessing import post_processing_example_1
from MorseGraphL4DC.analysis import full_morse_graph_analysis
from MorseGraphL4DC.plot import visualize_morse_sets_graph_basins, save_all_panels_individually
from MorseGraphL4DC.comparison import (compute_morse_set_iou_table, compute_roa_iou_table,
                                    compute_coverage_ratios, format_iou_tables_text)
from MorseGraphL4DC.cache import (save_method_results, load_method_results,
                               save_shared_data, load_shared_data, cache_exists)


# ==============================================================================
# System Parameters
# ==============================================================================

# Production parameters
L1, T1, U1 = 1.0, 3.0, 5.0
L2, T2, U2 = 1.0, 3.0, 5.0

# Degradation rates
gamma1, gamma2 = 1.00, 1.00

# Integration time (tau)
tau = 1.0

# Data generation parameters
N_trajectories = 50     # Number of initial trajectories
T_total = 10*tau        # Total integration time for each trajectory
random_seed = 42        # For reproducibility
# Expected data pairs: N_trajectories × (T_total/tau)


# ==============================================================================
# System Definition
# ==============================================================================

# Define toggle switch as SwitchingSystem
# Polynomials: p1(x) = x1 - T1, p2(x) = x2 - T2
polynomials = [
    lambda x: x[0] - T1,
    lambda x: x[1] - T2,
]

# 4 modes based on sign patterns: [sign(p1), sign(p2)]
# Mode 0: [-, -] (both low)  → high production
# Mode 1: [-, +] (p1 low, p2 high)
# Mode 2: [+, -] (p1 high, p2 low)
# Mode 3: [+, +] (both high) → low production
vector_fields = [
    lambda x: np.array([-gamma1*x[0] + U1, -gamma2*x[1] + U2]),  # Mode 0 [-, -]: b=[5, 5]
    lambda x: np.array([-gamma1*x[0] + U1, -gamma2*x[1] + L2]),  # Mode 1 [+, -]: b=[5, 1]
    lambda x: np.array([-gamma1*x[0] + L1, -gamma2*x[1] + U2]),  # Mode 2 [-, +]: b=[1, 5]
    lambda x: np.array([-gamma1*x[0] + L1, -gamma2*x[1] + L2]),  # Mode 3 [+, +]: b=[1, 1]
]

ode_system = SwitchingSystem(polynomials, vector_fields)


# ==============================================================================
# hat_f from Convex Optimization
# ==============================================================================

# Polynomial switching surfaces (from example1_sls_mode_polynomials.csv)
hat_polynomials = [
    lambda x: (-0.644627709090561
               + 0.196102360664951*x[0]
               + 0.0208041062738649*x[1]),  # f1 (linear)
    lambda x: (1.1572512694502
               - 0.000821512785661542*x[0]
               - 0.385939418892602*x[1]),  # f2 (linear)
]

# Vector fields (from example1_sls_vf_coeffs.csv)
# CSV modes 1,2,3,4 → binary modes 0,1,2,3
hat_vector_fields = [
    lambda x: np.array([-1.0*x[0] + 1.00000000000002,
                        -1.0*x[1] + 5.00000000000002]),  # Mode 0 (CSV mode 1): b=[1,5]
    lambda x: np.array([-1.0*x[0] + 1.00000000000001,
                        -1.0*x[1] + 1.00000000000001]),  # Mode 1 (CSV mode 2): b=[1,1]
    lambda x: np.array([-1.0*x[0] + 5.0,
                        -1.0*x[1] + 5.0]),               # Mode 2 (CSV mode 3): b=[5,5]
    lambda x: np.array([-1.0*x[0] + 5.00000000000002,
                        -1.0*x[1] + 1.0]),               # Mode 3 (CSV mode 4): b=[5,1]
]

hat_ode_system = SwitchingSystem(hat_polynomials, hat_vector_fields)


# ==============================================================================
# MorseGraph Parameters
# ==============================================================================

# Output directory
output_dir = f'example_1_tau_{tau:.1f}'.replace('.', '_')

# Phase space parameters
dim = 2
lower_bounds = np.array([0.0, 0.0])
upper_bounds = np.array([6.0, 6.0])

# Grid resolution : subdivide n times in each direction, so 2^7 x 2^7 boxes
subdivision = 7

# Plot labels (LaTeX-style subscripts)
xlabel = '$x_1$'
ylabel = '$x_2$'

# ==============================================================================
# Main Computation
# ==============================================================================

def main():
    """
    Computes Morse graphs for the toggle switch using 5 different F(rect) methods.

    Methods:
    1. F_integration (f): Numerical integration of the ground-truth system
    2. F_data: Data-driven approach with "outside" mapping for unmapped boxes
    3. F_gaussianprocess (GP): GP-based outer approximation with confidence bounds
    4. F_Lipschitz (hat_f): Lipschitz-based bounds
    5. F_integration (hat_f): Numerical integration where hat_f = f
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Toggle switch Morse graph analysis with caching')
    parser.add_argument('--force-recompute', nargs='*', type=int, metavar='N',
                       help='Methods to recompute (1-5). Example: --force-recompute 1 3 5')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching (compute but don\'t save)')
    args = parser.parse_args()

    # Configure caching
    force_methods = set(args.force_recompute or [])
    use_cache = not args.no_cache

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define Wong colorblind-safe palette for morse set visualization
    wong_colors = [
        (0.0, 0.45, 0.70),   # Blue
        (0.90, 0.60, 0.0),   # Orange
        (0.0, 0.62, 0.45),   # Bluish green
        (0.95, 0.90, 0.25),  # Yellow
        (0.0, 0.45, 0.70),   # Sky blue
        (0.80, 0.40, 0.0),   # Vermillion
        (0.80, 0.60, 0.70),  # Reddish purple
        (0.0, 0.0, 0.0)      # Black
    ]

    def apply_wong_colors(morse_graph):
        """Apply Wong colorblind-safe palette to morse sets."""
        morse_sets = list(morse_graph.nodes())
        for i, morse_set in enumerate(morse_sets):
            color = wong_colors[i % len(wong_colors)]
            morse_graph.nodes[morse_set]['color'] = color

    print("=" * 80)
    print("MorseGraph Example 1: Toggle Switch")
    print("Comparing 5 different F(rect) computation methods")
    print("=" * 80 + "\n")

    # Set random seeds for reproducibility
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    print(f"Random seed set to {random_seed} for reproducibility\n")

    # Set up common UniformGrid for all methods
    divisions = np.array([2**subdivision, 2**subdivision], dtype=int)
    grid = UniformGrid(bounds=np.array([lower_bounds, upper_bounds]),
                       divisions=divisions)

    # Compute padding parameter: grid cell diameter (used by some methods)
    box_size = (upper_bounds - lower_bounds) / (2**subdivision)
    epsilon = np.linalg.norm(box_size)

    print(f"Grid configuration:")
    print(f"  Resolution: {divisions[0]} × {divisions[1]} = {len(grid.get_boxes())} boxes\n")

    results_summary = []

    # =========================================================================
    # METHOD 1: F_integration - Ground-Truth f
    # =========================================================================
    print("=" * 80)
    print("METHOD 1: F_integration - Numerical ODE Integration (Ground-truth f)")
    print("=" * 80)

    # Check cache
    if 1 not in force_methods and cache_exists(output_dir, 1):
        print("  Loading from cache...")
        cached = load_method_results(output_dir, 1)
        box_map1 = cached['box_map']
        morse_graph1 = cached['morse_graph']
        basins1 = cached['basins']
        combined_morse_graph = cached['combined_morse_graph']
        combined_roas = cached['combined_roas']
        t_compute1 = cached['metadata']['computation_time']
        print(f"  Loaded from cache (saved {t_compute1:.2f}s of computation)")
    else:
        # Compute from scratch
        F1 = F_integration(ode_system, tau)

        # Compute Morse graph analysis
        t_start = time.time()
        box_map1, morse_graph1, basins1 = full_morse_graph_analysis(grid, F1)
        t_compute1 = time.time() - t_start
        print(f"  Computation time: {t_compute1:.2f} seconds")

        # Post-processing
        combined_morse_graph, combined_roas = post_processing_example_1(
            grid, box_map1, morse_graph1, basins1, T1, T2
        )

        # Save to cache
        if use_cache:
            save_method_results(output_dir, 1, box_map1, morse_graph1, basins1,
                              combined_morse_graph, combined_roas,
                              {'computation_time': t_compute1})
            print("  Saved to cache")

    # Apply colors for visualization
    apply_wong_colors(morse_graph1)

    # Visualize (individual panels + combined 3-panel)
    bounds = np.array([lower_bounds, upper_bounds])
    base_name = os.path.join(output_dir, 'method1_integration_f')
    save_all_panels_individually(
        grid, morse_graph1, basins1, box_map1,
        base_name, bounds,
        method_label='F_integration (f)',
        labels=(xlabel, ylabel),
        show_outside=False
    )

    # Create results dict
    results1 = {
        'method': 'F_integration (f)',
        'num_morse_sets': len(morse_graph1.nodes()),
        'num_edges': len(morse_graph1.edges()),
        'morse_set_box_counts': {i: len(ms) for i, ms in enumerate(morse_graph1.nodes())},
        'basin_box_counts': {i: len(basins1[ms]) for i, ms in enumerate(morse_graph1.nodes())},
        'computation_time': t_compute1
    }

    print(f"  Morse sets: {results1['num_morse_sets']}")
    print(f"  Edges: {results1['num_edges']}")

    # Add combined metrics
    results1['combined_morse_set_box_counts'] = {i: len(node) for i, node in enumerate(combined_morse_graph.nodes())}
    results1['combined_roa_box_counts'] = {i: len(combined_roas[node]) for i, node in enumerate(combined_morse_graph.nodes())}
    results1['combined_morse_graph'] = combined_morse_graph
    results1['combined_roas'] = combined_roas

    results_summary.append(results1)

    # Visualize combined morse graph (individual panels + combined 3-panel)
    base_name_combined = os.path.join(output_dir, 'method1_integration_f_combined')
    save_all_panels_individually(
        grid, combined_morse_graph, combined_roas, combined_morse_graph,
        base_name_combined, bounds,
        method_label='F_integration (f) - Combined',
        labels=(xlabel, ylabel),
        show_outside=False
    )

    # =========================================================================
    # DATA GENERATION
    # =========================================================================
    print("=" * 80)
    print("DATA GENERATION for F_data and F_gaussianprocess")
    print("=" * 80)

    # Check if shared data cache exists
    need_data_generation = (2 in force_methods or 4 in force_methods or
                           not cache_exists(output_dir, None))

    if not need_data_generation and cache_exists(output_dir, None):
        print("  Loading trajectory data from cache...")
        shared_data = load_shared_data(output_dir)
        trajectories = shared_data['trajectories']
        X = shared_data['X']
        Y = shared_data['Y']
        print(f"  Loaded {len(trajectories)} trajectories with X.shape={X.shape}, Y.shape={Y.shape}")
    else:
        print(f"  Sampling along {N_trajectories} trajectories with T_total={T_total}, tau={tau}")
        print(f"  Expected: ~{N_trajectories * int(T_total/tau)} data pairs from {N_trajectories} integrations")

        # Generate trajectory data
        trajectories = generate_trajectory_data(ode_system, tau, N_trajectories,
                                               np.array([lower_bounds, upper_bounds]),
                                               T_total=T_total, seed=random_seed)
        print(f"  Generated {len(trajectories)} trajectories")

        # Extract tau-map pairs: X[i] = φ(i*τ, x₀), Y[i] = φ((i+1)*τ, x₀)
        X = np.vstack([traj[:-1] for traj in trajectories])  # First n-1 points from each trajectory
        Y = np.vstack([traj[1:] for traj in trajectories])   # Last n-1 points from each trajectory
        print(f"  Extracted X, Y with shapes {X.shape}, {Y.shape}")

        # Save trajectory data (GP model will be saved separately in METHOD 4)
        if use_cache:
            save_shared_data(output_dir, trajectories, X, Y)
            print("  Saved trajectory data to cache")

    # Flatten all trajectory points for vector field evaluation (for .mat file)
    trajectory_data = np.vstack(trajectories)  # All points: φ(0,x), φ(τ,x), ..., φ(T_total,x)
    print(f"  Computing vector field at {len(trajectory_data)} trajectory points...")
    F_field = np.array([ode_system(0, x) for x in trajectory_data])
    print(f"  Computed vector field at {len(trajectory_data)} points")

    # Compute modes using binary encoding: mode = sum((p_i(x) > 0) * 2^i)
    print(f"  Computing mode classification for all trajectory points...")
    modes_f = compute_modes(trajectory_data, polynomials)
    modes_hat_f = compute_modes(trajectory_data, hat_polynomials)
    print(f"  Computed modes: f (analytical) and hat_f (learned)")

    # Save MATLAB-compatible data
    import scipy.io as sio
    mat_filename = os.path.join(output_dir, 'toggle_switch_data.mat')
    sio.savemat(mat_filename, {
        'trajectory_data': trajectory_data,  # All trajectory points
        'F_field': F_field,                  # Vector field f(x) at all trajectory points
        'modes_f': modes_f,                  # Mode indices for analytical system f
        'modes_hat_f': modes_hat_f           # Mode indices for learned system hat_f
    })
    print(f"  Saved data to: {mat_filename}")
    print(f"    - trajectory_data, F_field, modes_f, modes_hat_f ({len(trajectory_data)} points)")

    # =========================================================================
    # METHOD 2: F_data - Data-Driven
    # =========================================================================
    print("=" * 80)
    print("METHOD 2: F_data - Data-Driven")
    print("=" * 80)

    # Check cache
    if 2 not in force_methods and cache_exists(output_dir, 2):
        print("  Loading from cache...")
        cached = load_method_results(output_dir, 2)
        box_map2 = cached['box_map']
        morse_graph2 = cached['morse_graph']
        basins2 = cached['basins']
        combined_morse_graph2 = cached['combined_morse_graph']
        combined_roas2 = cached['combined_roas']
        t_compute2 = cached['metadata']['computation_time']

        print(f"  Loaded from cache (saved {t_compute2:.2f}s of computation)")
    else:
        # Compute from scratch
        print(f"  Using {len(X)} trajectory samples from shared dataset...")

        # Create F_data with "outside" mapping
        F2 = F_data(X, Y, grid, map_empty='outside')

        # Compute Morse graph analysis
        t_start = time.time()
        box_map2, morse_graph2, basins2 = full_morse_graph_analysis(grid, F2)
        t_compute2 = time.time() - t_start
        print(f"  Computation time: {t_compute2:.2f} seconds")

        # Post-processing
        combined_morse_graph2, combined_roas2 = post_processing_example_1(
            grid, box_map2, morse_graph2, basins2, T1, T2
        )

        # Save to cache
        if use_cache:
            save_method_results(output_dir, 2, box_map2, morse_graph2, basins2,
                              combined_morse_graph2, combined_roas2,
                              {'computation_time': t_compute2})
            print("  Saved to cache")

    # Apply colors for visualization
    apply_wong_colors(morse_graph2)

    # Visualize (individual panels + combined 3-panel)
    base_name = os.path.join(output_dir, 'method2_data')
    save_all_panels_individually(
        grid, morse_graph2, basins2, box_map2,
        base_name, bounds,
        method_label='F_data',
        labels=(xlabel, ylabel),
        show_outside=True  # F_data shows outside boxes
    )

    # Create results dict
    results2 = {
        'method': 'F_data',
        'num_morse_sets': len(morse_graph2.nodes()),
        'num_edges': len(morse_graph2.edges()),
        'morse_set_box_counts': {i: len(ms) for i, ms in enumerate(morse_graph2.nodes())},
        'basin_box_counts': {i: len(basins2[ms]) for i, ms in enumerate(morse_graph2.nodes())},
        'computation_time': t_compute2
    }

    print(f"  Morse sets: {results2['num_morse_sets']}")
    print(f"  Edges: {results2['num_edges']}")

    # Add combined metrics
    results2['combined_morse_set_box_counts'] = {i: len(node) for i, node in enumerate(combined_morse_graph2.nodes())}
    results2['combined_roa_box_counts'] = {i: len(combined_roas2[node]) for i, node in enumerate(combined_morse_graph2.nodes())}
    results2['combined_morse_graph'] = combined_morse_graph2
    results2['combined_roas'] = combined_roas2

    results_summary.append(results2)

    # Visualize combined morse graph (individual panels + combined 3-panel)
    base_name_combined = os.path.join(output_dir, 'method2_data_combined')
    save_all_panels_individually(
        grid, combined_morse_graph2, combined_roas2, combined_morse_graph2,
        base_name_combined, bounds,
        method_label='F_data - Combined',
        labels=(xlabel, ylabel),
        show_outside=False
    )

    # =========================================================================
    # METHOD 3: F_Lipschitz (f)
    # =========================================================================
    print("=" * 80)
    print("METHOD 3: F_Lipschitz (f) - Lipschitz-based Outer Approximation")
    print("=" * 80)

    # Check cache
    if 3 not in force_methods and cache_exists(output_dir, 3):
        print("  Loading from cache...")
        cached = load_method_results(output_dir, 3)
        box_map3 = cached['box_map']
        morse_graph3 = cached['morse_graph']
        basins3 = cached['basins']
        combined_morse_graph3 = cached['combined_morse_graph']
        combined_roas3 = cached['combined_roas']
        t_compute3 = cached['metadata']['computation_time']

        print(f"  Loaded from cache (saved {t_compute3:.2f}s of computation)")
    else:
        # Compute from scratch
        # Compute bound for Lipschitz constant
        L_tau = np.exp(-min(gamma1, gamma2) * tau)
        print(f"  Lipschitz constant: L_tau = exp(-gamma_min * tau) = {L_tau:.6f}")

        # F_Lipschitz padding: L_tau * (box_diameter/2)
        box_diameter = np.linalg.norm(box_size)
        padding_radius = L_tau * box_diameter / 2
        print(f"  Box diameter (grid cell diagonal): {box_diameter:.6f}")
        print(f"  Padding radius: L_tau * (diameter/2) = {padding_radius:.6f}")

        # Define tau_map and F_Lipschitz using ground-truth system
        # define_tau_map will auto-detect SwitchingSystem and use event detection
        tau_map = define_tau_map(ode_system, tau, max_step=0.05, rtol=1e-6, atol=1e-9)
        F3 = F_Lipschitz(tau_map, L_tau=L_tau, box_diameter=box_diameter)

        # Compute Morse graph analysis
        t_start = time.time()
        box_map3, morse_graph3, basins3 = full_morse_graph_analysis(grid, F3)
        t_compute3 = time.time() - t_start
        print(f"  Computation time: {t_compute3:.2f} seconds")

        # Post-processing
        combined_morse_graph3, combined_roas3 = post_processing_example_1(
            grid, box_map3, morse_graph3, basins3, T1, T2
        )

        # Save to cache
        if use_cache:
            save_method_results(output_dir, 3, box_map3, morse_graph3, basins3,
                              combined_morse_graph3, combined_roas3,
                              {'computation_time': t_compute3})
            print("  Saved to cache")

    # Apply colors for visualization
    apply_wong_colors(morse_graph3)

    # Visualize (individual panels + combined 3-panel)
    base_name = os.path.join(output_dir, 'method3_lipschitz_f')
    save_all_panels_individually(
        grid, morse_graph3, basins3, box_map3,
        base_name, bounds,
        method_label='F_Lipschitz (f)',
        labels=(xlabel, ylabel),
        show_outside=False
    )

    # Create results dict
    results3 = {
        'method': 'F_Lipschitz (f)',
        'num_morse_sets': len(morse_graph3.nodes()),
        'num_edges': len(morse_graph3.edges()),
        'morse_set_box_counts': {i: len(ms) for i, ms in enumerate(morse_graph3.nodes())},
        'basin_box_counts': {i: len(basins3[ms]) for i, ms in enumerate(morse_graph3.nodes())},
        'computation_time': t_compute3
    }

    print(f"  Morse sets: {results3['num_morse_sets']}")
    print(f"  Edges: {results3['num_edges']}")

    # Add combined metrics
    results3['combined_morse_set_box_counts'] = {i: len(node) for i, node in enumerate(combined_morse_graph3.nodes())}
    results3['combined_roa_box_counts'] = {i: len(combined_roas3[node]) for i, node in enumerate(combined_morse_graph3.nodes())}
    results3['combined_morse_graph'] = combined_morse_graph3
    results3['combined_roas'] = combined_roas3

    results_summary.append(results3)

    # Visualize combined morse graph (individual panels + combined 3-panel)
    base_name_combined = os.path.join(output_dir, 'method3_lipschitz_f_combined')
    save_all_panels_individually(
        grid, combined_morse_graph3, combined_roas3, combined_morse_graph3,
        base_name_combined, bounds,
        method_label='F_Lipschitz (f) - Combined',
        labels=(xlabel, ylabel),
        show_outside=False
    )

    # =========================================================================
    # METHOD 4: F_gaussianprocess (GP) - Data-Driven with Confidence Bounds
    # =========================================================================
    print("=" * 80)
    print("METHOD 4: F_gaussianprocess - GP-based Outer Approximation")
    print("=" * 80)

    # Check if GP model is cached
    shared_data = load_shared_data(output_dir)
    if shared_data is not None and shared_data['gp_model'] is not None and 4 not in force_methods:
        print("  Loading GP model from cache...")
        gp_model = shared_data['gp_model']
        gp_train_time = 0.0  # Already trained
        print("  GP model loaded from cache")
    else:
        # Train GP model
        print(f"  Using {len(X)} trajectory samples from dataset...")
        print(f"  Training GP model...")
        gp_start = time.time()
        gp_model = train_gp_from_data(X, Y, kernel_type='matern', nu=2.5)
        gp_train_time = time.time() - gp_start
        print(f"  GP training completed in {gp_train_time:.2f}s")

        # Save GP model to shared cache
        if use_cache:
            save_shared_data(output_dir, trajectories, X, Y, gp_model)
            print("  Saved GP model to cache")

    print(f"  Confidence level: 95% (1-δ = 0.95)")

    # Check method cache
    if 4 not in force_methods and cache_exists(output_dir, 4):
        print("  Loading Morse graph from cache...")
        cached = load_method_results(output_dir, 4)
        box_map4 = cached['box_map']
        morse_graph4 = cached['morse_graph']
        basins4 = cached['basins']
        combined_morse_graph4 = cached['combined_morse_graph']
        combined_roas4 = cached['combined_roas']
        t_compute4 = cached['metadata']['computation_time']

        print(f"  Loaded from cache (saved {t_compute4:.2f}s of computation)")
    else:
        # Compute from scratch
        # Create F_gaussianprocess with trained model
        F4 = F_gaussianprocess(gp_model, confidence_level=0.95, epsilon=0.0)

        # Compute Morse graph analysis
        t_start = time.time()
        box_map4, morse_graph4, basins4 = full_morse_graph_analysis(grid, F4)
        t_compute4 = time.time() - t_start
        print(f"  Computation time: {t_compute4:.2f} seconds")

        # Post-processing
        combined_morse_graph4, combined_roas4 = post_processing_example_1(
            grid, box_map4, morse_graph4, basins4, T1, T2
        )

        # Save to cache
        if use_cache:
            save_method_results(output_dir, 4, box_map4, morse_graph4, basins4,
                              combined_morse_graph4, combined_roas4,
                              {'computation_time': t_compute4, 'gp_train_time': gp_train_time})
            print("  Saved to cache")

    # Apply colors for visualization
    apply_wong_colors(morse_graph4)

    # Visualize (individual panels + combined 3-panel)
    base_name = os.path.join(output_dir, 'method4_gaussianprocess')
    save_all_panels_individually(
        grid, morse_graph4, basins4, box_map4,
        base_name, bounds,
        method_label='F_gaussianprocess (GP)',
        labels=(xlabel, ylabel),
        show_outside=False
    )

    # Create results dict
    results4 = {
        'method': 'F_gaussianprocess (GP)',
        'num_morse_sets': len(morse_graph4.nodes()),
        'num_edges': len(morse_graph4.edges()),
        'morse_set_box_counts': {i: len(ms) for i, ms in enumerate(morse_graph4.nodes())},
        'basin_box_counts': {i: len(basins4[ms]) for i, ms in enumerate(morse_graph4.nodes())},
        'gp_train_time': gp_train_time,
        'computation_time': t_compute4
    }

    print(f"  Morse sets: {results4['num_morse_sets']}")
    print(f"  Edges: {results4['num_edges']}\n")

    # Add combined metrics
    results4['combined_morse_set_box_counts'] = {i: len(node) for i, node in enumerate(combined_morse_graph4.nodes())}
    results4['combined_roa_box_counts'] = {i: len(combined_roas4[node]) for i, node in enumerate(combined_morse_graph4.nodes())}
    results4['combined_morse_graph'] = combined_morse_graph4
    results4['combined_roas'] = combined_roas4

    results_summary.append(results4)

    # Visualize combined morse graph (individual panels + combined 3-panel)
    base_name_combined = os.path.join(output_dir, 'method4_gaussianprocess_combined')
    save_all_panels_individually(
        grid, combined_morse_graph4, combined_roas4, combined_morse_graph4,
        base_name_combined, bounds,
        method_label='F_gaussianprocess (GP) - Combined',
        labels=(xlabel, ylabel),
        show_outside=False
    )

    # =========================================================================
    # METHOD 5: F_integration (hat_f)
    # =========================================================================
    print("=" * 80)
    print("METHOD 5: F_integration (hat_f) - Numerical Integration")
    print("=" * 80)

    # Check cache
    if 5 not in force_methods and cache_exists(output_dir, 5):
        print("  Loading from cache...")
        cached = load_method_results(output_dir, 5)
        box_map5 = cached['box_map']
        morse_graph5 = cached['morse_graph']
        basins5 = cached['basins']
        combined_morse_graph5 = cached['combined_morse_graph']
        combined_roas5 = cached['combined_roas']
        t_compute5 = cached['metadata']['computation_time']

        print(f"  Loaded from cache (saved {t_compute5:.2f}s of computation)")
    else:
        # Compute from scratch
        print(f"  Using data-driven system hat_f from CSV coefficients")

        F5 = F_integration(hat_ode_system, tau)

        # Compute Morse graph analysis
        t_start = time.time()
        box_map5, morse_graph5, basins5 = full_morse_graph_analysis(grid, F5)
        t_compute5 = time.time() - t_start
        print(f"  Computation time: {t_compute5:.2f} seconds")

        # Post-processing
        combined_morse_graph5, combined_roas5 = post_processing_example_1(
            grid, box_map5, morse_graph5, basins5, T1, T2
        )

        # Save to cache
        if use_cache:
            save_method_results(output_dir, 5, box_map5, morse_graph5, basins5,
                              combined_morse_graph5, combined_roas5,
                              {'computation_time': t_compute5})
            print("  Saved to cache")

    # Apply colors for visualization
    apply_wong_colors(morse_graph5)

    # Visualize (individual panels + combined 3-panel)
    base_name = os.path.join(output_dir, 'method5_integration_hatf')
    save_all_panels_individually(
        grid, morse_graph5, basins5, box_map5,
        base_name, bounds,
        method_label='F_integration (hat_f)',
        labels=(xlabel, ylabel),
        show_outside=False
    )

    # Create results dict
    results5 = {
        'method': 'F_integration (hat_f)',
        'num_morse_sets': len(morse_graph5.nodes()),
        'num_edges': len(morse_graph5.edges()),
        'morse_set_box_counts': {i: len(ms) for i, ms in enumerate(morse_graph5.nodes())},
        'basin_box_counts': {i: len(basins5[ms]) for i, ms in enumerate(morse_graph5.nodes())},
        'computation_time': t_compute5
    }

    print(f"  Morse sets: {results5['num_morse_sets']}")
    print(f"  Edges: {results5['num_edges']}")

    # Add combined metrics
    results5['combined_morse_set_box_counts'] = {i: len(node) for i, node in enumerate(combined_morse_graph5.nodes())}
    results5['combined_roa_box_counts'] = {i: len(combined_roas5[node]) for i, node in enumerate(combined_morse_graph5.nodes())}
    results5['combined_morse_graph'] = combined_morse_graph5
    results5['combined_roas'] = combined_roas5

    results_summary.append(results5)

    # Visualize combined morse graph (individual panels + combined 3-panel)
    base_name_combined = os.path.join(output_dir, 'method5_integration_hatf_combined')
    save_all_panels_individually(
        grid, combined_morse_graph5, combined_roas5, combined_morse_graph5,
        base_name_combined, bounds,
        method_label='F_integration (hat_f) - Combined',
        labels=(xlabel, ylabel),
        show_outside=False
    )

    # =========================================================================
    # Comparison Summary
    # =========================================================================
    print("=" * 80)
    print("COMPARISON SUMMARY - BEFORE POST-PROCESSING")
    print("=" * 80)
    print(f"{'Method':<30} {'MS':<6} {'Edges':<8} {'Avg MS Size':<12} {'Avg Basin':<12} {'Time (s)':<10}")
    print("-" * 80)
    for res in results_summary:
        avg_ms_size = np.mean(list(res['morse_set_box_counts'].values()))
        avg_basin = np.mean(list(res['basin_box_counts'].values()))
        comp_time = res.get('computation_time', 0.0)
        print(f"{res['method']:<30} {res['num_morse_sets']:<6} {res['num_edges']:<8} "
              f"{avg_ms_size:<12.1f} {avg_basin:<12.1f} {comp_time:<10.2f}")

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY - AFTER POST-PROCESSING")
    print("=" * 80)
    print(f"{'Method':<30} {'MS':<6} {'Avg MS Size':<12} {'Avg Basin':<12} {'Time (s)':<10}")
    print("-" * 80)
    for res in results_summary:
        num_collapsed = len(res['combined_morse_set_box_counts'])
        avg_collapsed_ms = np.mean(list(res['combined_morse_set_box_counts'].values()))
        avg_collapsed_basin = np.mean(list(res['combined_roa_box_counts'].values()))
        comp_time = res.get('computation_time', 0.0)
        print(f"{res['method']:<30} {num_collapsed:<6} {avg_collapsed_ms:<12.1f} {avg_collapsed_basin:<12.1f} {comp_time:<10.2f}")
    print()

    # =========================================================================
    # IoU COMPARISON (vs Ground Truth)
    # =========================================================================
    print("\n" + "=" * 80)
    print("IoU COMPARISON - Per-Node Analysis")
    print("=" * 80)

    # Collect combined graphs and basins
    combined_morse_graphs_list = [
        results_summary[0]['combined_morse_graph'],  # F_integration(f) - GT
        results_summary[1]['combined_morse_graph'],  # F_data
        results_summary[2]['combined_morse_graph'],  # F_Lipschitz
        results_summary[3]['combined_morse_graph'],  # F_gaussianprocess
        results_summary[4]['combined_morse_graph'],  # F_integration(hat_f)
    ]

    combined_roas_list = [
        results_summary[0]['combined_roas'],  # F_integration(f) - GT
        results_summary[1]['combined_roas'],  # F_data
        results_summary[2]['combined_roas'],  # F_Lipschitz
        results_summary[3]['combined_roas'],  # F_gaussianprocess
        results_summary[4]['combined_roas'],  # F_integration(hat_f)
    ]

    method_names = [res['method'] for res in results_summary]

    # Compute IoU tables
    morse_set_iou = compute_morse_set_iou_table(
        combined_morse_graphs_list,
        method_names,
        ground_truth_idx=0
    )

    roa_iou = compute_roa_iou_table(
        combined_roas_list,
        combined_morse_graphs_list,
        method_names,
        ground_truth_idx=0
    )

    coverage_ratios = compute_coverage_ratios(
        combined_morse_graphs_list,
        method_names,
        ground_truth_idx=0
    )

    # Format and print IoU tables
    iou_tables_text = format_iou_tables_text(
        morse_set_iou,
        roa_iou,
        coverage_ratios,
        ground_truth_name=method_names[0]
    )

    print(iou_tables_text)
    print()

    # Computation time summary
    print("=" * 80)
    print("COMPUTATION TIME COMPARISON")
    print("=" * 80)
    print(f"{'Method':<40} {'Time (s)':<15} {'Speedup vs F_int(f)':<20}")
    print("-" * 80)
    gt_time = results_summary[0]['computation_time']
    for res in results_summary:
        comp_time = res.get('computation_time', 0.0)
        speedup = gt_time / comp_time if comp_time > 0 else float('inf')
        speedup_str = f"{speedup:.2f}x" if speedup != float('inf') else "N/A"
        print(f"{res['method']:<40} {comp_time:<15.2f} {speedup_str:<20}")
    print()

    # =========================================================================
    # METRICS COMPARISON (vs Ground Truth)
    # =========================================================================
    from MorseGraphL4DC.metrics import compute_all_metrics

    print("\n" + "=" * 80)
    print("METRICS COMPARISON (vs Ground Truth - F_integration)")
    print("=" * 80)

    # Method 1 (F_integration) is Ground Truth
    gt_morse_graph = results_summary[0]['combined_morse_graph']
    gt_basins = results_summary[0]['combined_roas']

    metrics_table = []
    for i, result in enumerate(results_summary):
        if i == 0:  # Skip GT vs GT
            continue

        print(f"\nComputing metrics for {result['method']}...")
        metrics = compute_all_metrics(
            result['combined_morse_graph'],
            result['combined_roas'],
            gt_morse_graph,
            gt_basins,
            grid
        )

        metrics_table.append({
            'method': result['method'],
            'mean_iou': metrics['roa']['mean_iou'],
            'num_matched': metrics['roa']['num_matched'],
            'isomorphic': metrics['graph']['isomorphic'],
            'edit_distance': metrics['graph']['edit_distance'],
            'normalized_edit_distance': metrics['graph']['normalized_edit_distance']
        })

    # Print and save metrics table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    # Prepare table for both printing and saving
    table_lines = []
    table_lines.append("=" * 80)
    table_lines.append("METRICS COMPARISON (vs Ground Truth - F_integration)")
    table_lines.append("=" * 80)
    table_lines.append(f"{'Method':<30} {'IoU':<10} {'Matched':<10} {'Iso':<8} {'Edit Dst':<10}")
    table_lines.append("-" * 80)

    for m in metrics_table:
        iso_str = 'Yes' if m['isomorphic'] else 'No'
        edit_str = f"{m['normalized_edit_distance']:.3f}" if m['normalized_edit_distance'] is not None else 'N/A'
        line = f"{m['method']:<30} {m['mean_iou']:<10.4f} {m['num_matched']:<10} {iso_str:<8} {edit_str:<10}"
        table_lines.append(line)
        print(line)

    table_lines.append("=" * 80)
    print()

    # Save metrics table to .txt file (including IoU tables and timing)
    txt_output_path = os.path.join(output_dir, 'metrics_comparison.txt')
    with open(txt_output_path, 'w') as f:
        # Write IoU tables first
        f.write(iou_tables_text)
        f.write('\n\n')

        # Write computation time comparison
        f.write("=" * 80 + '\n')
        f.write("COMPUTATION TIME COMPARISON\n")
        f.write("=" * 80 + '\n')
        f.write(f"{'Method':<40} {'Time (s)':<15} {'Speedup vs F_int(f)':<20}\n")
        f.write("-" * 80 + '\n')
        for res in results_summary:
            comp_time = res.get('computation_time', 0.0)
            speedup = gt_time / comp_time if comp_time > 0 else float('inf')
            speedup_str = f"{speedup:.2f}x" if speedup != float('inf') else "N/A"
            f.write(f"{res['method']:<40} {comp_time:<15.2f} {speedup_str:<20}\n")
        f.write('\n\n')

        # Then write original metrics
        f.write('\n'.join(table_lines))

    # Save metrics to JSON (including IoU metrics)
    import json
    json_output_path = os.path.join(output_dir, 'metrics_comparison.json')

    # Combine all metrics into one JSON
    json_output = {
        'computation_times': {
            res['method']: res.get('computation_time', 0.0)
            for res in results_summary
        },
        'iou_metrics': {
            'morse_set_iou': morse_set_iou,
            'roa_iou': roa_iou,
            'coverage_ratios': coverage_ratios
        },
        'graph_metrics': metrics_table
    }

    with open(json_output_path, 'w') as f:
        json.dump(json_output, f, indent=2, default=str)

    print(f"Metrics saved to: {txt_output_path}")
    print(f"JSON data saved to: {json_output_path}\n")


if __name__ == "__main__":
    main()