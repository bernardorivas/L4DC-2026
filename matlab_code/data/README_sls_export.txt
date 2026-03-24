FILES:
  sls_identified_model.mat   - MATLAB struct with matrices, f(x), hard λ(x).
  sls_identified_model.json  - Same content in JSON format.
  sls_matrices.csv           - Identified A1, A2.
  sls_switch_surface.csv     - Polynomial coefficients and exponents.
  sls_full_data_with_modes.csv - Full dataset with f(x), hard labels, σ=±1.

SWITCHING LAW:
  lambda(x) = [1,0] if f(x)>=0; [0,1] if f(x)<0.
  The switching surface is f(x)=0, where f(x) = Σ a_j x1^{e1_j} x2^{e2_j}.
  In this experiment, f(x) ≈ x1^2 - 1.

POLYNOMIAL SURFACE ORDER:
  Exponents ordered by total degree 0..kappa, within each e1=0..total, e2=total-e1.
