FILES:
  sps_identified_model.mat   - MATLAB struct with poly dynamics, surface, hard λ(x).
  sps_identified_model.json  - Same content in JSON.
  sps_mode_polynomials.csv   - Coeffs (degree d) on normalized inputs.
  sps_mode_polynomials_RAW.csv - Dynamics expanded to RAW x (physical).
  sps_switch_surface.csv     - Surface coeffs (degree kappa) on normalized inputs.
  sps_switch_surface_RAW.csv - Surface expanded to RAW x.
  sps_full_data_with_modes.csv - Full dataset with f(x) and hard labels.

EVALUATION (RAW → NORM → PRED → RAW):
  x_norm = (x - mx)./sx;
  v_norm = [Φ_d(x_norm)*a_q, Φ_d(x_norm)*b_q];
  v_raw  = v_norm .* sd + md;

SWITCHING LAW:
  mode(x) = 1 if f((x-mx)./sx) >= 0; else 2.

NOTES:
  Exponent order matches buildMonomialMatrix (or the local generator below).
