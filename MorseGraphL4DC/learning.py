"""
Gaussian Process learning utilities for data-driven Morse graph computation.

Implements GP-based outer approximation methods from:
"Data-Driven Approximations of Dynamical Systems Operators for Control"
(arXiv:2210.01292v1)
"""

import numpy as np
from typing import Tuple, Optional, Callable
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import warnings


class GaussianProcessModel:
    """
    Gaussian Process model for approximating dynamical systems.

    Trains independent GPs for each output dimension, following the approach
    in Section 2.2 of arXiv:2210.01292v1.

    Uses Matérn kernel (ν=2.5) which ensures Lipschitz-continuous predictions,
    suitable for control applications.
    """

    def __init__(self, kernel_type='matern', nu=2.5, length_scale=1.0,
                 noise_level=1e-5, optimize_hyperparameters=True):
        """
        Initialize GP model.

        :param kernel_type: Type of kernel ('matern' or 'rbf')
        :param nu: Smoothness parameter for Matérn kernel (default: 2.5)
        :param length_scale: Initial length scale for kernel
        :param noise_level: Initial noise level (WhiteKernel)
        :param optimize_hyperparameters: Whether to optimize hyperparameters during fit
        """
        self.kernel_type = kernel_type
        self.nu = nu
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.optimize_hyperparameters = optimize_hyperparameters

        # Will be populated during fit
        self.gp_models = None
        self.input_dim = None
        self.output_dim = None
        self.X_train = None
        self.Y_train = None

    def _create_kernel(self, input_dim, X=None, Y=None):
        """
        Create kernel for GP with domain-aware length scale initialization.
        
        :param input_dim: Number of input dimensions
        :param X: Training inputs for computing data-aware length scales (optional)
        :param Y: Training outputs for computing output scale (optional)
        """
        # Compute initial length scales from data if available
        if X is not None and len(X) > 1:
            data_range = np.ptp(X, axis=0)  # peak-to-peak range per dimension
            # Use median data range for uniform scaling
            median_range = np.median(data_range)
            # Initialize to reasonable fraction of data range
            initial_length_scale = max(median_range / 10.0, 0.1)
            # Set bounds with safety margins
            length_scale_bounds = (max(median_range / 100.0, 1e-2), 
                                  max(median_range * 3.0, 10.0))
        else:
            # Fallback to default values
            initial_length_scale = self.length_scale
            length_scale_bounds = (1e-2, 1e2)
        
        if self.kernel_type == 'matern':
            # Matérn kernel with constant and white noise
            # k(x, x') = σ² * Matérn(x, x'; l, ν) + σ²_noise * δ(x, x')
            kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
                     Matern(length_scale=[initial_length_scale] * input_dim,
                           length_scale_bounds=length_scale_bounds,
                           nu=self.nu) +
                     WhiteKernel(noise_level=self.noise_level,
                                noise_level_bounds=(1e-10, 1e-1)))
        elif self.kernel_type == 'rbf':
            from sklearn.gaussian_process.kernels import RBF
            kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
                     RBF(length_scale=[initial_length_scale] * input_dim,
                        length_scale_bounds=length_scale_bounds) +
                     WhiteKernel(noise_level=self.noise_level,
                                noise_level_bounds=(1e-10, 1e-1)))
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        return kernel

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Train independent GPs for each output dimension.

        :param X: Training inputs, shape (N, D_in)
        :param Y: Training outputs, shape (N, D_out)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Validate minimum sample size for GP fitting
        if X.shape[0] < 2:
            raise ValueError(f"GP requires at least 2 training samples, got {X.shape[0]}")

        # Validate shapes match
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have same number of samples. Got X: {X.shape[0]}, Y: {Y.shape[0]}")

        self.X_train = X
        self.Y_train = Y
        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]

        # Train one GP per output dimension
        self.gp_models = []

        for i in range(self.output_dim):
            # Create kernel for this dimension
            kernel = self._create_kernel(self.input_dim, X, Y[:, i])

            # Create and fit GP
            # CRITICAL: normalize_y=False to match paper's formulation
            # Paper assumes predictions µ(x) are in original state space
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10 if self.optimize_hyperparameters else 0,
                alpha=0.0,  # Noise is handled by WhiteKernel
                normalize_y=False
            )

            # Suppress optimization warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X, Y[:, i])

            self.gp_models.append(gp)

        return self

    def predict(self, X: np.ndarray, return_std=False, return_cov=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance for input points.

        :param X: Input points, shape (N, D_in) or (D_in,) for single point
        :param return_std: If True, return standard deviation instead of variance
        :param return_cov: If True, return full covariance (not implemented, diagonal only)
        :return: (mean, variance) or (mean, std) depending on return_std
                 mean: shape (N, D_out) or (D_out,) for single point
                 variance/std: shape (N, D_out) or (D_out,) for single point
        """
        if self.gp_models is None:
            raise RuntimeError("Model must be fitted before prediction")

        # Handle single point
        single_point = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            single_point = True

        # Validate input dimensions
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input dimensions, got {X.shape[1]}")

        N = X.shape[0]

        # Predict for each output dimension
        means = np.zeros((N, self.output_dim))
        variances = np.zeros((N, self.output_dim))

        for i, gp in enumerate(self.gp_models):
            mean, std = gp.predict(X, return_std=True)
            means[:, i] = mean
            variances[:, i] = std**2

        # Return format
        if single_point:
            means = means.flatten()
            variances = variances.flatten()

        if return_std:
            return means, np.sqrt(variances)
        else:
            return means, variances

    def get_hyperparameters(self):
        """Get optimized hyperparameters for each GP."""
        if self.gp_models is None:
            raise RuntimeError("Model must be fitted first")

        return [gp.kernel_ for gp in self.gp_models]


def train_gp_from_data(X: np.ndarray, Y: np.ndarray,
                       kernel_type='matern', nu=2.5,
                       optimize_hyperparameters=True,
                       verbose=False) -> GaussianProcessModel:
    """
    Train a Gaussian Process model from trajectory data.

    :param X: Input states, shape (N, D_in)
    :param Y: Output states (after time-τ map), shape (N, D_out)
    :param kernel_type: Kernel type ('matern' or 'rbf')
    :param nu: Smoothness parameter for Matérn kernel
    :param optimize_hyperparameters: Whether to optimize hyperparameters
    :param verbose: If True, print diagnostic information about training
    :return: Fitted GaussianProcessModel
    """
    if verbose:
        print(f"  GP Training Diagnostics:")
        print(f"    Training samples: {X.shape[0]}")
        print(f"    Input dimensions: {X.shape[1]}")
        print(f"    Output dimensions: {Y.shape[1]}")
        print(f"    X range: [{np.min(X, axis=0)}, {np.max(X, axis=0)}]")
        print(f"    Y range: [{np.min(Y, axis=0)}, {np.max(Y, axis=0)}]")
    
    gp_model = GaussianProcessModel(
        kernel_type=kernel_type,
        nu=nu,
        optimize_hyperparameters=optimize_hyperparameters
    )

    gp_model.fit(X, Y)
    
    if verbose and optimize_hyperparameters:
        print(f"    Optimized hyperparameters:")
        for i, gp in enumerate(gp_model.gp_models):
            print(f"      Dimension {i}: {gp.kernel_}")

    return gp_model


