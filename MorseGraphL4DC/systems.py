"""
Standard dynamical systems for testing and examples.

This module provides canonical implementations of well-known dynamical systems
commonly used in testing and demonstrating Morse graph computations.

The SwitchingSystem class provides a structured representation of hybrid systems
that switch between multiple vector fields based on polynomial sign conditions.
"""

import numpy as np


class SwitchingSystem:
    """
    Switching system with mode assignment from polynomial sign patterns.

    Mathematical formulation:
        ẋ = f_σ(x)(x) = Σ_j λ_j(x) f_j(x)

    where:
        - σ(x) ∈ {0, ..., M-1} is the mode index
        - λ_j(x) ∈ {0,1} are binary indicators with Σ λ_j(x) = 1
        - f_j: R^n → R^n are the mode vector fields
        - Mode σ(x) determined by sign pattern of polynomials [p_1(x), ..., p_k(x)]
    """

    def __init__(self, polynomials, vector_fields):
        """
        Initialize switching system.

        Args:
            polynomials: List of callables [p_1, ..., p_k] where p_i: R^n → R
                        Mode determined by binary encoding of [sign(p_1), ..., sign(p_k)]
            vector_fields: List of callables [f_1, ..., f_M] where f_j: R^n → R^n
                          Ordered by sign pattern: M = 2^k modes for k polynomials
                          Encoding: sign(p_i) < 0 → 0, sign(p_i) > 0 → 1
        """
        self.polynomials = polynomials
        self.vector_fields = vector_fields
        self.n_polynomials = len(polynomials)
        self.n_modes = len(vector_fields)

        if self.n_modes != 2**self.n_polynomials:
            raise ValueError(f"Expected {2**self.n_polynomials} vector fields for "
                           f"{self.n_polynomials} polynomials, got {self.n_modes}")

    def sigma(self, x):
        """
        Compute mode index σ(x) ∈ {0, ..., M-1} from polynomial sign pattern.

        Binary encoding: [sign(p_1), ..., sign(k)] → integer
        Example: [-, +, -] → [0, 1, 0] → 0*1 + 1*2 + 0*4 = 2

        Args:
            x: State vector (numpy array or array-like)

        Returns:
            Mode index (int)
        """
        x = np.asarray(x)
        return sum((1 if p(x) > 0 else 0) * (2**i) for i, p in enumerate(self.polynomials))

    def lambda_vector(self, x):
        """
        Compute binary indicator vector [λ_1(x), ..., λ_M(x)].

        Exactly one entry is 1.0, all others are 0.0.

        Args:
            x: State vector (numpy array or array-like)

        Returns:
            Binary indicator vector (numpy array of length M)
        """
        lam = np.zeros(self.n_modes)
        lam[self.sigma(x)] = 1.0
        return lam

    def evaluate(self, x):
        """
        Evaluate vector field: f(x) = f_{σ(x)}(x).

        Args:
            x: State vector (numpy array or array-like)

        Returns:
            Vector field evaluation f(x) (numpy array)
        """
        x = np.asarray(x)
        idx = self.sigma(x)
        return self.vector_fields[idx](x)

    def is_on_switching_surface(self, x, tol=1e-10):
        """
        Check if x is on the switching surface.

        The switching surface is the union of zero sets: {x : p_i(x) = 0 for some i}.

        Args:
            x: State vector (numpy array or array-like)
            tol: Tolerance for zero checking (default: 1e-10)

        Returns:
            True if any polynomial is approximately zero at x
        """
        x = np.asarray(x)
        return any(abs(p(x)) < tol for p in self.polynomials)

    def __call__(self, t, x):
        """
        ODE signature for compatibility with scipy.integrate.solve_ivp.

        Args:
            t: Time (unused, but required by ODE solver interface)
            x: State vector

        Returns:
            ẋ = f(x)
        """
        return self.evaluate(x)
