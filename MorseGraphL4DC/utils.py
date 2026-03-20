"""
Utility functions for data generation and spatial operations.

This module provides helper functions for working with trajectory data
and spatial operations on Morse sets.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import warnings
from typing import Callable, Any

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# Tau-Map Data Generation
# =============================================================================

def generate_trajectory_data(ode_func, tau, N_trajectories, bounds, T_total=10, seed=None):
    """
    Generate trajectory data by sampling along ODE trajectories.

    Strategy: Sample N_trajectories initial conditions and integrate each for T_total time
    at time steps 0, tau, 2*tau, ..., T_total. Returns complete trajectories that can be
    used to extract tau-map pairs and vector field samples.

    This gives N_trajectories trajectories, each with (T_total/tau + 1) time points,
    providing efficient data generation for GP training and data-driven methods.

    :param ode_func: ODE function (t, x) -> dx/dt
    :param tau: Time step for τ-map
    :param N_trajectories: Number of initial trajectories
    :param bounds: Domain bounds array of shape (2, D)
    :param T_total: Total integration time (default: 10)
    :param seed: Random seed for reproducibility
    :return: List of trajectories, where each trajectory is array of shape (n_steps, dim)
             containing [φ(0,x), φ(τ,x), φ(2τ,x), ..., φ(T_total,x)]

    Example:
        >>> from MorseGraphL4DC.utils import generate_trajectory_data
        >>> from MorseGraphL4DC.systems import toggle_switch
        >>> from functools import partial
        >>> ode = partial(toggle_switch, L1=1, T1=3, U1=5, L2=1, T2=3, U2=5, gamma1=1, gamma2=1)
        >>> bounds = np.array([[0., 0.], [6., 6.]])
        >>> trajectories = generate_trajectory_data(ode, tau=1.0, N_trajectories=50, bounds=bounds, T_total=10)
        >>> print(f"Generated {len(trajectories)} trajectories, each with {len(trajectories[0])} points")
        >>> # Extract tau-map pairs: X = first n-1 points, Y = last n-1 points
        >>> X = np.vstack([traj[:-1] for traj in trajectories])
        >>> Y = np.vstack([traj[1:] for traj in trajectories])
    """
    if seed is not None:
        np.random.seed(seed)

    # Sample N_trajectories random initial conditions
    dim = bounds.shape[1]
    X0 = np.random.uniform(bounds[0], bounds[1], size=(N_trajectories, dim))

    # Time points to sample along each trajectory
    # Use linspace to avoid floating-point errors with arange
    n_steps = int(np.round(T_total / tau)) + 1
    t_eval = np.linspace(0, T_total, n_steps)

    trajectories = []
    failed_count = 0

    # Integrate each trajectory
    for x0 in X0:
        try:
            sol = solve_ivp(ode_func, [0, T_total], x0, method='RK45',
                          t_eval=t_eval, dense_output=False)

            # Store complete trajectory: φ(0,x), φ(τ,x), ..., φ(T_total,x)
            trajectory = sol.y.T  # Shape: (n_steps, dim)
            trajectories.append(trajectory)

        except Exception as e:
            failed_count += 1
            print(f"Warning: Integration failed for point {x0}: {e}")
            continue

    # Report summary
    if failed_count > 0:
        failure_rate = failed_count / N_trajectories
        print(f"  Skipped {failed_count}/{N_trajectories} trajectories ({failure_rate*100:.1f}%) due to integration failures")
        if failure_rate > 0.1:
            warnings.warn(f"Over 10% integration failures; results may be biased")

    return trajectories


# =============================================================================
# Tau-Map Construction
# =============================================================================

def define_tau_map(ode_f, tau: float, method: str = 'RK45',
                         rtol: float = 1e-6, atol: float = 1e-8,
                         max_step: float = None):
    """
    Create a point map for time-τ integration using scipy.integrate.solve_ivp.

    Automatically detects SwitchingSystem instances and enables event detection
    for precise switching surface crossing.

    Useful for creating the map_f argument for F_Lipschitz.

    :param ode_f: ODE right-hand side function f(t, x)
                 If ode_f is a SwitchingSystem instance, automatically
                 creates event functions for switching surface detection.
    :param tau: Integration time
    :param method: Integration method ('RK45', 'RK23', 'BDF', etc.)
    :param rtol: Relative tolerance
    :param atol: Absolute tolerance
    :param max_step: Maximum step size (for handling discontinuities)
    :return: map_f function that maps x -> φ_τ(x)

    Example:
        >>> def ode_rhs(t, x):
        ...     return np.array([-x[0] + 1, -x[1] + 2])
        >>> map_f = define_tau_map(ode_rhs, tau=0.1)
        >>> x_next = map_f(np.array([0.5, 0.5]))
    """
    # Auto-detect SwitchingSystem and create event functions
    event_functions = []
    from MorseGraphL4DC.systems import SwitchingSystem
    if isinstance(ode_f, SwitchingSystem):
        # Create event functions for each polynomial (switching surface)
        for poly in ode_f.polynomials:
            def event_func(t, x, p=poly):
                return p(x)
            event_func.terminal = False
            event_functions.append(event_func)

    def map_f(x):
        """Evaluate φ_τ(x) via ODE integration."""
        kwargs = dict(method=method, rtol=rtol, atol=atol)
        if max_step is not None:
            kwargs['max_step'] = max_step
        
        # Add event functions if they exist (for SwitchingSystem)
        if event_functions:
            kwargs['events'] = event_functions

        sol = solve_ivp(ode_f, [0.0, tau], x, **kwargs)

        if not sol.status == 0:
            warnings.warn(f"ODE integration warning: {sol.message}")

        return sol.y[:, -1]

    return map_f


# =============================================================================
# Morse Set Spatial Operations
# =============================================================================

def find_box_containing_point(grid, point):
    """
    Find the box index containing the given point.

    :param grid: UniformGrid object
    :param point: Point coordinates (numpy array or list)
    :return: Box index (int)
    """
    # For UniformGrid, compute box indices from point coordinates
    lower_bounds = grid.bounds[0]
    box_widths = (grid.bounds[1] - grid.bounds[0]) / grid.divisions

    # Compute which box the point falls into
    indices = np.floor((np.array(point) - lower_bounds) / box_widths).astype(int)

    # Clamp to valid range
    indices = np.clip(indices, 0, grid.divisions - 1)

    # Convert multi-dimensional index to linear box index
    # For 2D: box_idx = i * divisions[1] + j
    box_idx = indices[0] * grid.divisions[1] + indices[1]

    return box_idx


def compute_morse_set_centroid(grid, morse_set):
    """
    Compute the centroid of all boxes in a morse set.

    :param grid: UniformGrid object
    :param morse_set: Frozenset of box indices
    :return: Centroid coordinates (numpy array)
    """
    box_centers = []
    for box_idx in morse_set:
        # Convert box index to multi-dimensional indices
        indices = np.unravel_index(box_idx, grid.divisions)

        # Compute box center
        lower_bounds = grid.bounds[0]
        box_widths = (grid.bounds[1] - grid.bounds[0]) / grid.divisions
        center = lower_bounds + (np.array(indices) + 0.5) * box_widths
        box_centers.append(center)

    # Compute centroid as mean of all box centers
    return np.mean(box_centers, axis=0)

def compute_modes(trajectory_data, polynomials):
    """
    Compute mode for each point using binary conversion.
    
    Mode encoding: mode = sum_{i=0}^{k-1} (p_i(x) > 0) * 2^i
    
    This maps polynomial sign patterns to mode indices:
    - All polynomials negative → mode 0
    - p_0 positive, rest negative → mode 1  
    - p_1 positive, rest negative → mode 2
    - p_0 and p_1 positive → mode 3
    - etc.
    
    :param trajectory_data: Array of shape (N, D) with trajectory points
    :param polynomials: List of polynomial functions [p_0, ..., p_{k-1}]
    :return: Array of shape (N,) with integer mode indices
    
    Example:
        >>> polynomials = [lambda x: x[0] - 3, lambda x: x[1] - 3]
        >>> data = np.array([[2, 2], [4, 2], [2, 4], [4, 4]])
        >>> modes = compute_modes(data, polynomials)
        >>> # modes = [0, 1, 2, 3] for the four quadrants
    """
    import numpy as np
    
    N = len(trajectory_data)
    modes = np.zeros(N, dtype=int)
    
    for i, p in enumerate(polynomials):
        for j, x in enumerate(trajectory_data):
            if p(x) > 0:
                modes[j] += 2**i
    
    return modes


def matlab_mode_to_binary(matlab_mode: int, num_polynomials: int = None) -> int:
    """
    Convert MATLAB mode indexing to binary mode indexing.
    
    MATLAB convention: q = 1 + i + 2j + 4k + ... where i,j,k ∈ {0,1}
    Binary convention: mode = i + 2j + 4k + ...
    
    Conversion: binary_mode = matlab_mode - 1
    
    :param matlab_mode: 1-indexed mode from MATLAB/CSV (1, 2, 3, 4, ...)
    :param num_polynomials: Number of switching polynomials (for validation)
    :return: 0-indexed binary mode (0, 1, 2, 3, ...)
    
    Examples:
        >>> matlab_mode_to_binary(1, num_polynomials=2)  # [-, -] → mode 0
        0
        >>> matlab_mode_to_binary(4, num_polynomials=2)  # [+, -] → mode 3
        3
    """
    if matlab_mode < 1:
        raise ValueError(f"MATLAB mode must be >= 1, got {matlab_mode}")
    
    binary_mode = matlab_mode - 1
    
    # Optional validation
    if num_polynomials is not None:
        max_mode = 2**num_polynomials - 1
        if binary_mode > max_mode:
            raise ValueError(
                f"Binary mode {binary_mode} exceeds max for {num_polynomials} "
                f"polynomials (max = {max_mode})"
            )
    
    return binary_mode


def binary_mode_to_matlab(binary_mode: int) -> int:
    """
    Convert binary mode indexing to MATLAB mode indexing.
    
    Inverse of matlab_mode_to_binary.
    
    :param binary_mode: 0-indexed binary mode
    :return: 1-indexed MATLAB mode
    """
    return binary_mode + 1


def create_polynomial_from_coeffs(coeff_dict: dict, dim: int) -> callable:
    """
    Create polynomial lambda function from coefficient dictionary.
    
    Builds polynomial of form: sum(coeff * x[0]^e1 * x[1]^e2 * ...)
    
    :param coeff_dict: Dictionary mapping exponent tuples to coefficients
                      Example: {(0,0): -10, (1,0): 5, (0,1): 3, (2,0): 2}
                      represents: -10 + 5*x[0] + 3*x[1] + 2*x[0]^2
    :param dim: State space dimension
    :return: Callable polynomial function
    
    Example:
        >>> coeffs = {(0,0): -10.0, (1,0): 5.0, (0,1): 3.0, (2,0): 2.0}
        >>> poly = create_polynomial_from_coeffs(coeffs, dim=2)
        >>> poly(np.array([1.0, 2.0]))  # -10 + 5*1 + 3*2 + 2*1^2 = 3.0
        3.0
    """
    import numpy as np
    
    # Capture coefficients in closure
    coeffs = dict(coeff_dict)  # Copy to avoid mutation
    
    def polynomial(x):
        """Evaluate polynomial at point x."""
        result = 0.0
        for exponents, coeff in coeffs.items():
            # Compute x[0]^e1 * x[1]^e2 * ...
            term = coeff
            for i, exp in enumerate(exponents):
                if exp > 0:
                    term *= x[i]**exp
            result += term
        return result
    
    return polynomial
