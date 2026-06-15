# Topological Dynamics via Learned Hybrid Systems

Python implementation for analyzing switching systems using a basic implementation of CMGDB

## Overview

This project computes Morse graphs, Morse sets, and regions of attraction for dynamical systems using:
- Box subdivision methods on uniform grids
- Time-τ maps derived from ODE integration
- Combinatorial Conley-Morse theory for dynamics

## Installation

### Prerequisites
- Python 3.8 or higher

### Setup

1. Clone the repository:
```bash
git clone https://github.com/bernardorivas/MorseGraphACC26.git
cd MorseGraphACC26
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
MorseGraphACC26/
├── MorseGraphL4DC/          # Main package
│   ├── core.py          # Box map and Morse graph computation
│   ├── grids.py         # Grid subdivision (uniform/adaptive)
│   ├── dynamics.py      # ODE dynamics and tau-maps
│   ├── systems.py       # Predefined dynamical systems
│   ├── analysis.py      # Analysis utilities
│   ├── utils.py         # Data generation and utilities
│   ├── plot.py          # Visualization functions
│   └── learning.py      # GP-based learning methods
├── examples/            # Usage examples
│   ├── example1.py      # Toggle switch system
│   └── example2.py      # Piecewise Van der Pol system
└── matlab_code/         # MATLAB for Convex Optimization algorithms
```

## Usage

### Running Examples

The project includes two examples:

```bash
# Example 1: Toggle switch system
python examples/example1.py

# Example 2: Piecewise Van der Pol oscillator
python examples/example2.py
```

### Basic Workflow

```python
import numpy as np
from MorseGraphL4DC.grids import UniformGrid
from MorseGraphL4DC.dynamics import F_integration
from MorseGraphL4DC.analysis import full_morse_graph_analysis

# Define ODE system: dx/dt = f(t, x)
def ode_system(t, x):
    return np.array([...])

# Create grid
grid = UniformGrid(
    bounds=np.array([[x_min, y_min], [x_max, y_max]]),
    divisions=np.array([nx, ny])
)

# Define dynamics using tau-map
F = F_integration(ode_system, tau=1.0, epsilon=0.1)

# Compute Morse graph
morse_graph = full_morse_graph_analysis(grid, F, max_iterations=5)
```

## Key Components

### Dynamics Methods

The package supports multiple approaches for computing tau-maps:

- **`F_integration`**: Direct ODE integration using scipy
- **`F_Lipschitz`**: Rigorous interval arithmetic bounds
- **`F_data`**: Data-driven KD-tree based maps
- **`F_gaussianprocess`**: GP regression for learned dynamics

### Grid Types

- **`UniformGrid`**: Regular subdivision of phase space

## Dependencies

- **numpy**: Numerical computations
- **scipy**: ODE integration and scientific computing
- **matplotlib**: Plotting and visualization
- **networkx**: Graph operations for Morse graphs
- **joblib**: Parallel computing
- **scikit-learn**: Gaussian process regression

## License

MIT License
