"""
MorseGraph: Computational topology for dynamical systems.

This library provides tools for computing and analyzing Morse graphs of
dynamical systems through discrete abstractions on grids.
"""

__version__ = "0.1.0"

# Core components
from .grids import AbstractGrid, UniformGrid, AdaptiveGrid
from .dynamics import (
    Dynamics,
    F_function,
    F_data,
    F_integration,
    F_Lipschitz
)
from .core import Model
from .analysis import (
    compute_morse_graph,
    compute_all_morse_set_basins
)
from .plot import (
    plot_morse_sets,
    plot_basins_of_attraction,
    plot_morse_graph,
    plot_morse_graph_academic
)
from .metrics import (
    compute_all_metrics,
    compute_roa_accuracy,
    compute_graph_metrics,
    compare_homology
)

# Utility modules
from . import systems
from . import utils

__all__ = [
    # Core
    'AbstractGrid',
    'UniformGrid',
    'AdaptiveGrid',
    'Dynamics',
    'F_function',
    'F_data',
    'F_integration',
    'F_Lipschitz',
    'Model',
    # Analysis
    'compute_morse_graph',
    'compute_all_morse_set_basins',
    # Plotting
    'plot_morse_sets',
    'plot_basins_of_attraction',
    'plot_morse_graph',
    'plot_morse_graph_academic',
    # Metrics
    'compute_all_metrics',
    'compute_roa_accuracy',
    'compute_graph_metrics',
    'compare_homology',
    # Modules
    'systems',
    'utils',
]

