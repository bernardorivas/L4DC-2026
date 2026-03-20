# Computational Topology for Dynamical Systems: Method Summary

## Overview

This project implements **data-driven Morse graph computation** for analyzing global dynamics of nonlinear systems. The framework compares five different methods for constructing outer approximations of the time-τ map, which underpins the computation of Morse decompositions and regions of attraction.

**Core Idea:** Given a dynamical system ẋ = f(x), discretize the state space into a grid, compute a multivalued map F that outer-approximates the time-τ flow, and extract topological features (Morse sets, basins of attraction) from the resulting directed graph.

---

## The Five Methods

### Method 1: F_integration(f) - Ground Truth ODE Integration

**Mathematical Formulation:**
- Given ODE: ẋ = f(t, x)
- Time-τ map: φ_τ(x) = solution at time t = τ starting from x
- For grid box ξ with vertices V(ξ):
  ```
  F(ξ) = BoundingBox({φ_τ(v) : v ∈ V(ξ)}) ⊕ B(0, ε)
  ```
- Bloating parameter: ε = 0.0 (no additional padding)

**Computational Algorithm:**
1. Sample all 2^D corners plus center of each grid box
2. Integrate ODE from each sample point using scipy's `solve_ivp`
3. For SwitchingSystem: enable event detection for switching surfaces
4. Compute axis-aligned bounding box of all image points
5. Find all grid cells intersecting the bounding box

**Implementation:** `MorseGraphL4DC.dynamics.F_integration`
```python
class F_integration(Dynamics):
    def __init__(self, ode_f, tau, epsilon=0.0)
    # Auto-detects SwitchingSystem and creates event functions
```

**Integration Parameters:**
- Method: RK45 (adaptive Runge-Kutta-Fehlberg)
- `rtol=1e-6`: relative tolerance
- `atol=1e-9`: absolute tolerance
- `max_step=0.05`: maximum step size (important for switching surfaces)
- Parallelization: `joblib` with `n_jobs=-1`

**Key Characteristics:**
- Most accurate (serves as ground truth)
- Computationally expensive (~40-90 seconds)
- Handles discontinuous switching surfaces with event detection
- Deterministic, reproducible results

---

### Method 2: F_data - Data-Driven Mapping

**Mathematical Formulation:**
- Given trajectory data: {(X_i, Y_i)} where Y_i = φ_τ(X_i)
- For grid box ξ:
  ```
  F(ξ) = BoundingBox({Y_i : X_i ∈ B(ξ, ε_in)}) ⊕ B(0, ε_out)
  ```
- Input bloating: ε_in = box_size (per dimension)
- Output bloating: ε_out = box_size (per dimension)
- Empty boxes: map to "outside" domain

**Computational Algorithm:**
1. Build spatial index (KD-tree) of data points X_i
2. For each grid box ξ:
   - Query points within ε_in distance (L1 or L2 norm)
   - Collect corresponding output points Y_i
   - Compute bounding box of Y_i with ε_out padding
   - If no data: assign "outside" label
3. Return all intersecting grid cells

**Implementation:** `MorseGraphL4DC.dynamics.F_data`
```python
class F_data(Dynamics):
    def __init__(self, X_data, Y_data, epsilon_in, epsilon_out,
                 metric='l1', enclosure='box_enclosure', map_empty='outside')
```

**Parameters:**
- `metric='l1'`: distance metric (Manhattan distance)
- `enclosure='box_enclosure'`: bounding box method
- `map_empty='outside'`: uncovered regions mapped outside

**Key Characteristics:**
- Very fast (~0.1-0.4 seconds, 200-300× speedup)
- Data-dependent coverage (may miss regions)
- Creates "outside" Morse set for unmapped areas
- No model training required

---

### Method 3: F_Lipschitz(f) - Lipschitz-Based Outer Approximation

**Mathematical Formulation:**
- Lipschitz condition: ‖φ_τ(x) - φ_τ(y)‖ ≤ L_τ ‖x - y‖
- For grid box ξ with diameter d:
  ```
  F(ξ) = ⋃_{v ∈ V(ξ)} B(φ_τ(v), r)
  where r = L_τ · (d/2) + ε
  ```
- d = diagonal length of box
- ε = additional bloating (typically 0)

**Lipschitz Constant Computation:**

**Example 1 (Toggle Switch):**
- Linear dynamics: ẋ = -γx + b
- Lipschitz constant: L_τ = exp(-γ_min · τ)
- With γ_min = min(γ_1, γ_2) = 1.0, τ = 1.0:
  ```
  L_τ = exp(-1.0) ≈ 0.3679
  ```

**Example 2 (Van der Pol):**
- Computed analytically from Jacobian bounds
- L_τ ≈ 20.64 (conservative estimate)

**Computational Algorithm:**
1. Compute Lipschitz constant L_τ (system-dependent)
2. Compute padding radius: r = L_τ · (box_diameter / 2)
3. For each box:
   - Extract all 2^D vertices
   - Apply point map φ_τ to each vertex (via ODE integration)
   - Compute bounding box of images plus padding r
4. Return all intersecting grid cells

**Implementation:** `MorseGraphL4DC.dynamics.F_Lipschitz`
```python
class F_Lipschitz(Dynamics):
    def __init__(self, map_f, L_tau, epsilon=0.0)
    # map_f: point map φ_τ
    # L_tau: Lipschitz constant
```

**Key Characteristics:**
- Rigorous outer approximation (provably correct)
- Requires analytical knowledge of L_τ
- Conservative (often larger than ground truth)
- Fast (only vertex evaluations, ~30-165 seconds)
- Quality depends on tightness of L_τ estimate

---

### Method 4: F_gaussianprocess - GP-Based Outer Approximation

**Mathematical Formulation:** (from Ewerton et al. arXiv:2210.01292v1)
- Train Gaussian Process: (μ(x), Σ(x)) from data {(X_i, Y_i)}
- For grid box ξ with center c:
  ```
  F(ξ) = R(ξ) ∪ E^δ_Σ(c)

  R(ξ) = BoundingBox({μ(v) : v ∈ V(ξ)})
  E^δ_Σ(c) = {y : ‖y - μ(c)‖_Σ^-1 ≤ z_{α/2}}
  ```
- Confidence region: μ(c) ± z_{α/2} · σ(c) per dimension
- Bonferroni correction: α = 1 - (1-δ)^(1/D) for D dimensions

**Computational Algorithm:**
1. **Training Phase** (one-time cost):
   - Train independent GP for each output dimension
   - Kernel: Matérn (ν=2.5) + Constant + WhiteKernel
   - Hyperparameter optimization with 10 restarts
   - Data-aware length scale initialization

2. **Prediction Phase** (per box):
   - Predict μ(v), σ(v) at all vertices → compute R(ξ)
   - Predict μ(c), σ(c) at center → compute E^δ_Σ(c)
   - Take union: min/max bounds across both regions
   - Return all intersecting grid cells

**Implementation:** `MorseGraphL4DC.learning.GaussianProcessModel`
```python
class GaussianProcessModel:
    def __init__(self, X_train, Y_train, kernel_type='matern',
                 nu=2.5, confidence_level=0.95)
```

**Kernel Structure:**
```python
Matérn(length_scale, nu=2.5)
  + ConstantKernel(1.0)
  + WhiteKernel(noise_level)
```

**Parameters:**
- `confidence_level=0.95`: 95% confidence intervals
- `nu=2.5`: Matérn smoothness parameter
- Noise bounds: [1e-5, 1e-1] (auto-tuned)
- Optimizer: L-BFGS-B with 10 random restarts

**Key Characteristics:**
- Probabilistic guarantees (not deterministic)
- Data-efficient (works with sparse samples)
- Training overhead (~5-30 seconds total)
- Smooth approximation (no "outside" regions)
- Best for smooth dynamics

---

### Method 5: F_integration(hat_f) - Learned Model Integration

**Mathematical Formulation:**
- Learned system: ẋ = ĥat_f(t, x) from convex optimization
- ĥat_f obtained via SOS (Sum-of-Squares) programming
- Same outer approximation as Method 1, but with learned dynamics

**Learned Model Structure:**

**Example 1 (Toggle Switch):**
- Learned from:
  - `examples/matlab_data/example1_sls_mode_polynomials.csv`
  - `examples/matlab_data/example1_sls_vf_coeffs.csv`
- 2 switching polynomials (linear)
- 4 mode vector fields (linear)
- SwitchingLinearSystem

**Example 2 (Van der Pol):**
- Learned from:
  - `examples/matlab_data/example2_sps_mode_polynomials.csv`
  - `examples/matlab_data/example2_sps_vf_coeffs.csv`
- 1 switching polynomial (quadratic)
- 2 mode vector fields (polynomial)
- SwitchingPolynomialSystem

**Computational Algorithm:**
- Identical to F_integration(f)
- Same event detection for switching surfaces
- Same integration parameters
- Different input: ĥat_ode_system instead of ode_system

**Implementation:** Same `F_integration` class, different input
```python
dynamics_hatf = F_integration(hat_ode_system, tau=1.0, epsilon=0.0)
```

**Key Characteristics:**
- Tests learned model accuracy
- Same computational cost as Method 1 (~40-50 seconds)
- Reveals approximation errors in system identification
- Requires pre-trained model from convex optimization

---

## The Two Examples

### Example 1: Toggle Switch (Bistability)

**System Equations:**
```
ẋ₁ = -γ₁·x₁ + b₁(x)
ẋ₂ = -γ₂·x₂ + b₂(x)

Production rates b₁(x), b₂(x):
  Mode 0 [x₁<T₁, x₂<T₂]: b = [U₁, U₂] = [5, 5]
  Mode 1 [x₁<T₁, x₂≥T₂]: b = [U₁, L₂] = [5, 1]
  Mode 2 [x₁≥T₁, x₂<T₂]: b = [L₁, U₂] = [1, 5]
  Mode 3 [x₁≥T₁, x₂≥T₂]: b = [L₁, L₂] = [1, 1]

Switching surfaces:
  p₁(x) = x₁ - T₁ = x₁ - 3.0
  p₂(x) = x₂ - T₂ = x₂ - 3.0
```

**Parameters:**
- Production: L₁ = L₂ = 1.0, U₁ = U₂ = 5.0
- Thresholds: T₁ = T₂ = 3.0
- Degradation: γ₁ = γ₂ = 1.0
- Domain: [0, 6] × [0, 6]
- Time step: τ = 1.0

**Grid Configuration:**
- Subdivision: 2^7 = 128 divisions per dimension
- Total boxes: 128² = 16,384 boxes
- Box size: 6/128 ≈ 0.0469 per dimension
- Box diagonal: 0.0469√2 ≈ 0.0663

**Data Generation:**
- Number of trajectories: N = 50
- Total integration time: T_total = 10·τ = 10.0
- Time points per trajectory: 11 (at 0, τ, 2τ, ..., 10τ)
- Total (X, Y) pairs: 50 × 10 = 500
- Random seed: 42
- File: `examples/example_1_tau_1_0/toggle_switch_data.mat`

**Expected Morse Structure:**
- **3 Morse sets** (after post-processing):
  - **Node 0**: Attractor at high-x₁, low-x₂ (approximately (5, 1))
  - **Node 1**: Attractor at low-x₁, high-x₂ (approximately (1, 5))
  - **Node 2**: Repeller/saddle region near (T₁, T₂) = (3, 3)
- **Morse graph edges**: 2→0, 2→1 (repeller flows to both attractors)

**Learned Model:**
- Source: `examples/matlab_data/example1_sls_*.csv`
- Type: Switching Linear System (SLS)
- Learned polynomials: 2 linear switching surfaces
- Learned vector fields: 4 linear modes
- Mode encoding: binary (mode = Σᵢ (pᵢ(x)>0)·2^i)

**Output Files:** (in `examples/example_1_tau_1_0/`)
```
toggle_switch_data.mat              # Trajectory data, modes
method1_integration_f_*.png         # Ground truth visualizations
method2_data_*.png                  # Data-driven results
method3_lipschitz_f_*.png          # Lipschitz method
method4_gaussianprocess_*.png      # GP method
method5_integration_hatf_*.png     # Learned model
method*_combined.png               # 3-panel: morse sets + graph + basins
method*_combined_*.png             # Post-processed 3-node structure
metrics_comparison.json            # Quantitative comparison
metrics_comparison.txt             # Human-readable tables
```

**Key Characteristics:**
- Bistable system with two stable equilibria
- Symmetric structure (by parameter choice)
- Classic test case for basin computation
- 4 modes from 2 switching surfaces

---

### Example 2: Piecewise Van der Pol (Limit Cycle)

**System Equations:**
```
ẋ₁ = x₂
ẋ₂ = μ(1 - x₁²)x₂ - g(x₁)

Piecewise function g(x₁):
  Mode 0 [|x₁| < 1]: g(x₁) = x₁³     (cubic restoring force)
  Mode 1 [|x₁| ≥ 1]: g(x₁) = x₁      (linear restoring force)

Switching polynomial:
  p(x) = x₁² - 1
```

**Parameters:**
- Van der Pol parameter: μ = 1.0
- Domain: [-3, 3] × [-3, 3]
- Time step: τ = 1.0

**Grid Configuration:**
- Subdivision: 2^7 = 128 divisions per dimension
- Total boxes: 128² = 16,384 boxes
- Box size: 6/128 ≈ 0.0469 per dimension
- Box diagonal: 0.0469√2 ≈ 0.0663

**Data Generation:**
- Number of trajectories: N = 50
- Total integration time: T_total = 10·τ = 10.0
- Time points per trajectory: 11
- Total (X, Y) pairs: 500
- Random seed: 42
- File: `examples/example_2_tau_1_0/piecewise_vdp_data.mat`

**Expected Morse Structure:**
- **2 Morse sets** (after post-processing):
  - **Node 0**: Stable limit cycle (attractor)
  - **Node 1**: Unstable region near origin (repeller)
- **Morse graph edge**: 1→0 (origin flows to limit cycle)

**Learned Model:**
- Source: `examples/matlab_data/example2_sps_*.csv`
- Type: Switching Polynomial System (SPS)
- Learned polynomial: 1 quadratic switching surface
- Learned vector fields: 2 polynomial modes
- Mode 0: x₂, -x₀ - x₀³ - x₀²x₁ + x₁
- Mode 1: x₂, -x₀ - x₀²x₁ + x₁

**Output Files:** (in `examples/example_2_tau_1_0/`)
```
piecewise_vdp_data.mat             # Trajectory data, modes
method1_integration_f_*.png        # Ground truth visualizations
method2_data_*.png                 # Data-driven results
method3_lipschitz_f_*.png         # Lipschitz method
method4_gaussianprocess_*.png     # GP method
method5_integration_hatf_*.png    # Learned model
method*_combined.png              # 3-panel: morse sets + graph + basins
method*_combined_*.png            # Post-processed 2-node structure
metrics_comparison.json           # Quantitative comparison
metrics_comparison.txt            # Human-readable tables
```

**Key Characteristics:**
- Limit cycle attractor (periodic orbit)
- Piecewise modification of classical Van der Pol
- Single switching surface: |x₁| = 1
- Tests periodic orbit capture
- 2 modes from 1 switching surface

---

## Computational Workflow

### Complete Pipeline (for each example):

**1. System Setup**
```python
# Define ground truth ODE
ode_system = SwitchingSystem(polynomials, vector_fields)

# Define learned model (from CSV)
hat_ode_system = SwitchingSystem(hat_polynomials, hat_vector_fields)

# Create uniform grid
grid = UniformGrid(bounds, subdivisions=[2**7, 2**7])
```

**2. Data Generation** (shared by Methods 2 & 4)
```python
# Sample initial conditions
trajectories = generate_trajectory_data(
    ode_func=ode_system,
    tau=1.0,
    N_trajectories=50,
    bounds=bounds,
    T_total=10.0,
    seed=42
)

# Extract (X, Y) pairs for τ-map
X = np.vstack([traj[:-1] for traj in trajectories])
Y = np.vstack([traj[1:] for traj in trajectories])

# Compute modes (for each point)
modes_f = compute_modes(X, polynomials)
modes_hatf = compute_modes(X, hat_polynomials)

# Save to MATLAB format
scipy.io.savemat('data.mat', {
    'X': X, 'Y': Y,
    'modes_f': modes_f,
    'modes_hatf': modes_hatf
})
```

**3. Method Computation** (×5 methods)
```python
# Initialize dynamics
dynamics = {
    'method1': F_integration(ode_system, tau=1.0),
    'method2': F_data(X, Y, epsilon_in, epsilon_out),
    'method3': F_Lipschitz(map_f, L_tau),
    'method4': GaussianProcessModel(X, Y).to_dynamics(grid),
    'method5': F_integration(hat_ode_system, tau=1.0)
}

# For each method:
for name, dyn in dynamics.items():
    # Compute box map (parallelized)
    box_map = Model(grid, dyn).compute()

    # Extract Morse graph
    morse_graph = compute_morse_graph(box_map)

    # Compute basins of attraction
    basins = compute_all_morse_set_basins(box_map, morse_graph)

    # Time the computation
    time_elapsed = ...
```

**4. Post-Processing**
```python
# Collapse Morse graph to canonical structure
morse_graph_combined, basins_combined = post_processing_function(
    box_map, morse_graph, basins
)

# Apply colorblind-safe colors
colors = colorblind_friendly_colors(n_nodes)
```

**5. Visualization**
```python
# Individual panels
save_morse_sets_only(grid, morse_graph, box_map, 'morse_sets.png')
save_morse_graph_only(morse_graph, 'morse_graph.png')
save_basins_only(grid, basins, morse_graph, 'basins.png')

# Combined 3-panel figure
save_all_panels_individually(
    grid, morse_graph, basins, box_map,
    base_name='method1_integration_f',
    bounds=bounds
)

# Post-processed visualizations
visualize_morse_sets_graph_basins(
    grid, morse_graph_combined, basins_combined, box_map,
    output_path='method1_integration_f_combined.png'
)
```

**6. Comparison Metrics**
```python
# Compute IoU tables
morse_iou = compute_morse_set_iou_table(
    morse_graphs, method_names, ground_truth_idx=0
)
basin_iou = compute_basin_iou_table(
    basins_list, morse_graphs, method_names, ground_truth_idx=0
)
coverage = compute_coverage_ratios(
    morse_graphs, method_names, ground_truth_idx=0
)

# Graph comparison
isomorphic = is_isomorphic(graph1, graph2)
edit_distance = normalized_edit_distance(graph1, graph2)

# Save results
save_metrics_json('metrics_comparison.json', all_metrics)
save_metrics_text('metrics_comparison.txt', all_metrics)
```

---

## Comparison Metrics

### Computed Metrics (in `MorseGraphL4DC.comparison` and `MorseGraphL4DC.metrics`)

**1. Morse Set IoU** (Intersection over Union)
```
IoU(M_i^method, M_i^GT) = |M_i^method ∩ M_i^GT| / |M_i^method ∪ M_i^GT|
```
- Per-node matching between methods and ground truth
- Measures spatial overlap of Morse sets
- Range: [0, 1], where 1 = perfect match
- Mean IoU: average across all nodes

**2. Basin IoU**
```
IoU(B_i^method, B_i^GT) = |B_i^method ∩ B_i^GT| / |B_i^method ∪ B_i^GT|
```
- Per-basin matching (includes Morse sets)
- Measures region of attraction accuracy
- Range: [0, 1], where 1 = perfect match
- Mean IoU: average across all basins

**3. Spatial Coverage Ratio**
```
Coverage(M_i) = |M_i^method| / |M_i^GT|
```
- Volume ratio indicating approximation quality
- Ratio > 1: over-approximation (conservative)
- Ratio < 1: under-approximation (aggressive)
- Ratio = 0: node not found by method

**4. Graph Isomorphism**
```
Isomorphic(G_method, G_GT) ∈ {True, False}
```
- Are Morse graphs structurally identical?
- Checks node count and edge connectivity
- Does NOT require spatial matching

**5. Graph Edit Distance**
```
EditDist(G_method, G_GT) = (node_ops + edge_ops) / max_ops
```
- Normalized distance between graph structures
- Range: [0, 1], where 0 = isomorphic
- Counts node/edge insertions and deletions

**6. Computation Time**
- Wall-clock time for Morse graph computation
- Includes: box map + SCC extraction + basins
- Excludes: data generation and visualization
- Speedup vs. ground truth: time_GT / time_method

### Output Format

**JSON** (`metrics_comparison.json`):
```json
{
  "method_name": {
    "morse_iou": {"node_0": 0.888, "node_1": 1.0, "mean": 0.944},
    "basin_iou": {"node_0": 0.835, "node_1": 0.996, "mean": 0.916},
    "coverage": {"node_0": 0.89, "node_1": 1.0},
    "time_seconds": 46.41,
    "speedup": 1.90,
    "isomorphic": true,
    "edit_distance": 0.0
  }
}
```

**TXT** (`metrics_comparison.txt`):
```
================================================================================
MORSE SET IoU vs F_integration (f)
================================================================================
                    F_data              F_Lipschitz (f)     F_gaussianprocess
--------------------------------------------------------------------------------
Node 0 (Attractor)  0.143               0.444               0.108
Node 1 (Attractor)  0.250               0.444               0.066
Mean IoU            0.141               0.630               0.067
...
```

---

## Key Implementation Classes

### 1. **SwitchingSystem** (`MorseGraphL4DC/systems.py`)

Hybrid dynamical system with polynomial switching surfaces.

```python
class SwitchingSystem:
    def __init__(self, polynomials: List[Callable],
                 vector_fields: List[Callable])

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray
    # Returns dx/dt based on active mode

    def get_mode(self, x: np.ndarray) -> int
    # Binary mode encoding: mode = Σᵢ (pᵢ(x)>0)·2^i
```

**Features:**
- Event-based integration (precise switching detection)
- Binary mode encoding
- Callable as ODE right-hand side

---

### 2. **UniformGrid** (`MorseGraphL4DC/grids.py`)

Uniform subdivision of state space.

```python
class UniformGrid:
    def __init__(self, bounds: np.ndarray,
                 subdivisions: List[int])

    def box_index(self, point: np.ndarray) -> int
    # Map point → box index (with caching)

    def box_bounds(self, index: int) -> np.ndarray
    # Get [lower, upper] bounds of box

    def dilate(self, box_indices: Set[int],
               radius: int = 1) -> Set[int]
    # Morphological dilation (neighbors)
```

**Features:**
- Fast point-to-box lookup with caching
- Supports arbitrary dimensions
- Neighbor operations (dilation)

---

### 3. **Dynamics Classes** (`MorseGraphL4DC/dynamics.py`)

All inherit from abstract `Dynamics` base class.

```python
class Dynamics(ABC):
    @abstractmethod
    def __call__(self, box: np.ndarray) -> np.ndarray:
        # Returns [lower, upper] bounding box

    def get_active_boxes(self, grid: UniformGrid) -> Set[int]:
        # Optional: filter boxes (e.g., exclude "outside")
```

**Implementations:**
- `F_integration`: ODE integration
- `F_data`: Data-driven lookup
- `F_Lipschitz`: Lipschitz bounds
- `F_gaussianprocess`: GP predictions (wrapper)
- Custom implementations possible

---

### 4. **Model** (`MorseGraphL4DC/core.py`)

Computes box map from grid + dynamics.

```python
class Model:
    def __init__(self, grid: UniformGrid,
                 dynamics: Dynamics)

    def compute(self, parallel: bool = True,
                n_jobs: int = -1) -> BoxMap
    # Returns dict: box_index → set of successor indices
```

**Features:**
- Parallelized computation (joblib)
- Progress tracking (tqdm)
- Handles "outside" boxes

---

### 5. **GaussianProcessModel** (`MorseGraphL4DC/learning.py`)

Wraps sklearn GP for dynamics prediction.

```python
class GaussianProcessModel:
    def __init__(self, X_train: np.ndarray,
                 Y_train: np.ndarray,
                 kernel_type: str = 'matern',
                 nu: float = 2.5,
                 confidence_level: float = 0.95)

    def train(self, n_restarts: int = 10)
    # Train independent GPs per dimension

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    # Returns (mean, std_dev)

    def to_dynamics(self, grid: UniformGrid,
                    epsilon: float = 0.0) -> Dynamics
    # Convert to F_gaussianprocess callable
```

**Kernel:**
```python
Matérn(length_scale, nu=2.5)
  + ConstantKernel(1.0)
  + WhiteKernel(noise_level)
```

---

### 6. **Analysis Functions** (`MorseGraphL4DC/analysis.py`)

Core topological computations.

```python
def compute_morse_graph(box_map: BoxMap,
                        compute_map: bool = True) -> MorseGraph
    # Extract SCCs + transitive reduction

def compute_all_morse_set_basins(
    box_map: BoxMap,
    morse_graph: MorseGraph,
    method: str = 'topological_priority'
) -> Dict[int, Set[int]]
    # Compute basins via backward reachability

def full_morse_graph_analysis(
    grid: UniformGrid,
    dynamics: Dynamics,
    parallel: bool = True
) -> Tuple[BoxMap, MorseGraph, Basins]
    # Complete pipeline: box map → morse graph → basins
```

---

## Parameters Summary

### Integration Parameters (Methods 1, 3, 5)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `method` | `'RK45'` | Adaptive Runge-Kutta-Fehlberg |
| `rtol` | `1e-6` | Relative tolerance |
| `atol` | `1e-9` | Absolute tolerance |
| `max_step` | `0.05` | Maximum step size |
| `t_eval` | `[tau]` | Evaluation times |
| `events` | `[event_funcs]` | Switching surface detection (auto) |
| `n_jobs` | `-1` | Parallel jobs (all CPUs) |

### Data Generation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_trajectories` | `50` | Number of initial conditions |
| `T_total` | `10.0` | Total integration time |
| `tau` | `1.0` | Time step for τ-map |
| `n_steps` | `11` | Time points: 0, τ, ..., 10τ |
| `seed` | `42` | Random seed |
| Total samples | `500` | N × n_steps = 50 × 10 |

### Grid Parameters

| Parameter | Example 1 | Example 2 |
|-----------|-----------|-----------|
| Domain | [0, 6]² | [-3, 3]² |
| Subdivisions | `[128, 128]` | `[128, 128]` |
| Total boxes | 16,384 | 16,384 |
| Box size | 0.0469 | 0.0469 |
| Box diagonal | 0.0663 | 0.0663 |

### Method-Specific Parameters

**F_data:**
- `epsilon_in` = box_size (per dimension)
- `epsilon_out` = box_size (per dimension)
- `metric` = 'l1' (Manhattan distance)
- `enclosure` = 'box_enclosure'
- `map_empty` = 'outside'

**F_Lipschitz:**
- Example 1: `L_tau` = exp(-1.0) ≈ 0.3679
- Example 2: `L_tau` = 20.64
- `epsilon` = 0.0

**F_gaussianprocess:**
- `kernel_type` = 'matern'
- `nu` = 2.5 (Matérn smoothness)
- `confidence_level` = 0.95
- `n_restarts` = 10 (hyperparameter optimization)
- Noise bounds: [1e-5, 1e-1]

---

## Differences Between Examples

| Aspect | Example 1: Toggle Switch | Example 2: Van der Pol |
|--------|-------------------------|------------------------|
| **System Type** | Piecewise linear | Piecewise polynomial |
| **Behavior** | Bistable (2 attractors) | Limit cycle (periodic orbit) |
| **Morse Sets** | 3 (2 attractors + 1 repeller) | 2 (1 attractor + 1 repeller) |
| **Domain** | [0, 6]² | [-3, 3]² |
| **Switching Surfaces** | 2 linear (x₁=3, x₂=3) | 1 quadratic (x₁²=1) |
| **Modes** | 4 (binary: 2 surfaces) | 2 (binary: 1 surface) |
| **Lipschitz L_τ** | exp(-1.0) ≈ 0.37 | 20.64 |
| **Learned Model** | SLS (linear) | SPS (polynomial) |
| **CSV Files** | `example1_sls_*.csv` | `example2_sps_*.csv` |
| **Post-processing** | `post_processing_example_1` | `post_processing_example_2` |
| **Output Prefix** | `toggle_switch_*` | `piecewise_vdp_*` |
| **Best Method** | F_integration(hat_f) | F_gaussianprocess |
| **Best IoU** | 0.835 (basins) | 0.574 (basins) |

---

## Output Files Reference

### Per-Method Files (for each of 5 methods)

**Pattern:** `method{1-5}_{name}_*.png`

Where `{name}` is:
- `integration_f` (Method 1)
- `data` (Method 2)
- `lipschitz_f` (Method 3)
- `gaussianprocess` (Method 4)
- `integration_hatf` (Method 5)

**Individual Panels:**
```
method{N}_{name}_morse_sets.png       # Morse sets colored by node
method{N}_{name}_morse_graph.png      # Directed graph (Hasse diagram)
method{N}_{name}_basins.png           # Basins of attraction
method{N}_{name}.png                  # Combined 3-panel (before post-processing)
```

**Post-Processed Visualizations:**
```
method{N}_{name}_combined_morse_sets.png
method{N}_{name}_combined_morse_graph.png
method{N}_{name}_combined_basins.png
method{N}_{name}_combined.png         # Combined 3-panel (after post-processing)
```

### Data Files

```
{system}_data.mat                     # Trajectory data
├── X              # Input points (N×D)
├── Y              # Output points (N×D)
├── modes_f        # Modes from ground truth (N×1)
└── modes_hatf     # Modes from learned model (N×1)
```

### Comparison Files

```
metrics_comparison.json               # Structured metrics (JSON)
metrics_comparison.txt                # Formatted tables (human-readable)
```

---

## Example Usage

### Running Example 1

```bash
cd examples
python example1.py
```

**Output:** All visualizations and metrics in `examples/example_1_tau_1_0/`

### Running Example 2

```bash
cd examples
python example2.py
```

**Output:** All visualizations and metrics in `examples/example_2_tau_1_0/`

### Custom System

```python
from MorseGraph import *
import numpy as np

# Define ODE
def my_ode(t, x):
    return np.array([x[1], -x[0]])  # Harmonic oscillator

# Create grid
bounds = np.array([[-2, -2], [2, 2]])
grid = UniformGrid(bounds, subdivisions=[64, 64])

# Choose dynamics
dynamics = F_integration(my_ode, tau=0.5)

# Compute Morse graph
box_map, morse_graph, basins = full_morse_graph_analysis(
    grid, dynamics, parallel=True
)

# Visualize
visualize_morse_sets_graph_basins(
    grid, morse_graph, basins, box_map,
    output_path='my_system.png'
)
```

---

## References

**Key Papers:**
- Conley Index Theory: Conley (1978)
- Computational Topology: Kaczynski, Mischaikow, Mrozek (2004)
- Data-Driven Methods: Bogdan et al. (2020), Yim et al. (2025)
- GP-Based Outer Approximation: Ewerton et al. (2022) arXiv:2210.01292v1
- Switching System Identification: Kaito et al. (2025)

**Code:** https://github.com/[your-repo]

---

**Last Updated:** 2025-11-05
