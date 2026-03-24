function [sol, prog, cost_vec, vars] = searchPoly(X, Xdot, lambda, prog_base, phi_d, opts)
% polynomial system id via 1-norm cost (2-mode)
% Identifies coefficients a1,b1,a2,b2 in vector fields using monomial basis phi_d.
%
% Adds three model-agnostic stabilizers:
%   (A) Jacobian L1 penalty (controls stiffness / Lipschitz)
%   (B) Degree- and sample-weighted objectives (reduces extrapolation)
%   (C) Optional first-order collocation on short windows (mitigates compounding)
%
% Inputs:
%   X      : N x 2 states [x,y]
%   Xdot   : N x 2 velocities [xdot, ydot]
%   lambda : N x 2 soft/hard mode weights
%   prog_base : Spotless base program
%   phi_d  : N x P monomial basis evaluated at X
%   opts   : (optional) struct with fields:
%       .lambda_coeff_L1   (default 0)     L1 penalty on coefficients
%       .l1_weights        (default 1)     P-vector weights for L1 on coefs (e.g., degree-weighted)
%       .lambda_dx_y       (default 0)     penalty enforcing xdot ≈ y (physics tether)
%       .eta_box           (default 10)    box bound on UN-SCALED coefficients
%       .scale_columns     (default true)  normalize columns of phi_d (+ same scaling for derivative bases)
%       .verbose           (default 12)    solver verbosity
%       .sample_weights    (default 1)     N-vector weights per sample for residuals
%       .lambda_jac        (default 0)     L1 penalty weight on Jacobian entries
%       .phi_dx, .phi_dy   (optional)      N x P derivative bases ∂phi/∂x, ∂phi/∂y (required if lambda_jac>0)
%       .lambda_colloc     (default 0)     L1 penalty weight for collocation windows
%       .colloc_pairs      (optional)      K x 2 indices [i, j] with dt between samples
%       .colloc_dt         (optional)      scalar or K-vector time step for each pair
%
% Outputs:
%   sol, prog, cost_vec, vars (includes unscaled coefficients)
%
% Notes:
% - All added penalties preserve convexity (linear + |·| slacks).
% - If you enable lambda_jac>0, you must supply phi_dx, phi_dy matching phi_d.

if nargin < 6, opts = struct; end
if ~isfield(opts,'lambda_coeff_L1'), opts.lambda_coeff_L1 = 0; end
if ~isfield(opts,'l1_weights'),      opts.l1_weights      = []; end
if ~isfield(opts,'lambda_dx_y'),     opts.lambda_dx_y     = 0; end
if ~isfield(opts,'eta_box'),         opts.eta_box         = 10; end
if ~isfield(opts,'scale_columns'),   opts.scale_columns   = true; end
if ~isfield(opts,'verbose'),         opts.verbose         = 12; end
if ~isfield(opts,'sample_weights'),  opts.sample_weights  = []; end
if ~isfield(opts,'lambda_jac'),      opts.lambda_jac      = 0; end
if ~isfield(opts,'lambda_colloc'),   opts.lambda_colloc   = 0; end
if ~isfield(opts,'colloc_pairs'),    opts.colloc_pairs    = []; end
if ~isfield(opts,'colloc_dt'),       opts.colloc_dt       = []; end

[N, P] = size(phi_d);
prog = prog_base;

% ---------- Optional column scaling for conditioning ----------
scale = ones(P,1);              % numeric, column
if opts.scale_columns
    scale = vecnorm(phi_d,2,1).';
    scale(scale==0) = 1;
    Phi = phi_d ./ scale.';     % N x P (numeric)
else
    Phi = phi_d;
end

% derivative bases for Jacobian (if requested)
use_jac = opts.lambda_jac > 0;
if use_jac
    if ~isfield(opts,'phi_dx') || ~isfield(opts,'phi_dy') || isempty(opts.phi_dx) || isempty(opts.phi_dy)
        error('searchPoly: lambda_jac>0 requires opts.phi_dx and opts.phi_dy.');
    end
    if opts.scale_columns
        Phi_x = opts.phi_dx ./ scale.';   % N x P
        Phi_y = opts.phi_dy ./ scale.';   % N x P
    else
        Phi_x = opts.phi_dx;
        Phi_y = opts.phi_dy;
    end
end

% sample weights for residuals
if isempty(opts.sample_weights), opts.sample_weights = ones(N,1); end
opts.sample_weights = opts.sample_weights(:);
if numel(opts.sample_weights) ~= N
    error('searchPoly: sample_weights must be N x 1.');
end

% degree weights for L1 on coefficients
if isempty(opts.l1_weights), opts.l1_weights = ones(P,1); end
opts.l1_weights = opts.l1_weights(:);
if numel(opts.l1_weights) ~= P
    error('searchPoly: l1_weights must be P x 1.');
end

% decision variables: coefficients in the *scaled* basis
[prog, a1_s] = prog.newFree(P); [prog, b1_s] = prog.newFree(P);
[prog, a2_s] = prog.newFree(P); [prog, b2_s] = prog.newFree(P);

% per-sample residuals for data fit (stacked in order x,y,x,y,...)
cost_vec = [];
% optional residuals for xdot ≈ y (physics tether)
dx_y_res = [];
% Jacobian |·| slack collector
J_slacks = [];

for i = 1:N
    phi_i = Phi(i, :)';         % P x 1 numeric
    lam1  = lambda(i,1);
    lam2  = lambda(i,2);

    % mode fields (scaled-basis coefs)
    f1 = [a1_s' * phi_i; b1_s' * phi_i];   % [xdot; ydot] under mode 1
    f2 = [a2_s' * phi_i; b2_s' * phi_i];   % mode 2
    fx = lam1 * f1 + lam2 * f2;

    residual = Xdot(i, :)' - fx;           % 2x1 msspoly
    cost_vec = [cost_vec; residual];

    if opts.lambda_dx_y > 0
        xdot_pred = fx(1);
        dx_y_res  = [dx_y_res; (X(i,2) - xdot_pred)];
    end

    if use_jac
        phix_i = Phi_x(i,:)';  % P x 1
        phiy_i = Phi_y(i,:)';  % P x 1
        % Jacobian entries (each linear in coefficients)
        J11 = lam1*(a1_s' * phix_i) + lam2*(a2_s' * phix_i);
        J12 = lam1*(a1_s' * phiy_i) + lam2*(a2_s' * phiy_i);
        J21 = lam1*(b1_s' * phix_i) + lam2*(b2_s' * phix_i);
        J22 = lam1*(b1_s' * phiy_i) + lam2*(b2_s' * phiy_i);
        % |·| slacks
        [prog, s11] = prog.newPos(1); prog = prog.withPos( s11 - J11 ); prog = prog.withPos( s11 + J11 );
        [prog, s12] = prog.newPos(1); prog = prog.withPos( s12 - J12 ); prog = prog.withPos( s12 + J12 );
        [prog, s21] = prog.newPos(1); prog = prog.withPos( s21 - J21 ); prog = prog.withPos( s21 + J21 );
        [prog, s22] = prog.newPos(1); prog = prog.withPos( s22 - J22 ); prog = prog.withPos( s22 + J22 );
        J_slacks = [J_slacks; s11; s12; s21; s22];
    end
end

% 1-norm slacks for data residuals (2N entries)
[prog, delta] = prog.newPos(length(cost_vec));
prog = prog.withPos(delta - cost_vec);
prog = prog.withPos(delta + cost_vec);

% Optional 1-norm slacks for xdot ≈ y
if opts.lambda_dx_y > 0
    [prog, delta_dx] = prog.newPos(length(dx_y_res));
    prog = prog.withPos(delta_dx - dx_y_res);
    prog = prog.withPos(delta_dx + dx_y_res);
else
    delta_dx = 0; % numeric zero is fine in objective
end

% Collocation penalties (optional)
colloc_terms = [];
if opts.lambda_colloc > 0 && ~isempty(opts.colloc_pairs)
    pairs = opts.colloc_pairs;
    if size(pairs,2) ~= 2, error('searchPoly: colloc_pairs must be Kx2 indices.'); end
    K = size(pairs,1);
    dt = opts.colloc_dt;
    if isempty(dt), error('searchPoly: provide colloc_dt (scalar or K-vector).'); end
    if isscalar(dt), dt = dt * ones(K,1); end
    if numel(dt) ~= K, error('searchPoly: colloc_dt length must match #pairs.'); end

    for k = 1:K
        i  = pairs(k,1);
        j  = pairs(k,2);
        hk = dt(k);

        % f(X_i)
        lam1  = lambda(i,1);  lam2 = lambda(i,2);
        phi_i = Phi(i,:)';
        f1_i  = [a1_s' * phi_i; b1_s' * phi_i];
        f2_i  = [a2_s' * phi_i; b2_s' * phi_i];
        f_i   = lam1*f1_i + lam2*f2_i;   % 2x1

        % First-order consistency: X_j ≈ X_i + h * f(X_i)
        colloc_res = (X(j,:)' - (X(i,:)' + hk * f_i));  % 2x1 msspoly

        % |·| slacks
        [prog, s_c] = prog.newPos(2);
        prog = prog.withPos( s_c - colloc_res );
        prog = prog.withPos( s_c + colloc_res );
        colloc_terms = [colloc_terms; s_c];
    end
end

% L1 coefficient regularization via |coef|
ua1=0; ub1=0; ua2=0; ub2=0;  % default numeric zeros
if opts.lambda_coeff_L1 > 0
    [prog, ua1] = prog.newPos(P); [prog, ub1] = prog.newPos(P);
    [prog, ua2] = prog.newPos(P); [prog, ub2] = prog.newPos(P);

    prog = prog.withPos(ua1 - a1_s); prog = prog.withPos(ua1 + a1_s);
    prog = prog.withPos(ub1 - b1_s); prog = prog.withPos(ub1 + b1_s);
    prog = prog.withPos(ua2 - a2_s); prog = prog.withPos(ua2 + a2_s);
    prog = prog.withPos(ub2 - b2_s); prog = prog.withPos(ub2 + b2_s);
end

% --------- BOX CONSTRAINTS (msspoly-safe) ----------
% We want |a|<=eta, |b|<=eta in UN-SCALED space, with a = a_s ./ scale.
% Equivalent: |a_s| <= eta * scale, |b_s| <= eta * scale  (elementwise).
eta = opts.eta_box;
eta_scaled = eta * scale;                % numeric P x 1
prog = prog.withPos( eta_scaled - a1_s );  prog = prog.withPos( eta_scaled + a1_s );
prog = prog.withPos( eta_scaled - b1_s );  prog = prog.withPos( eta_scaled + b1_s );
prog = prog.withPos( eta_scaled - a2_s );  prog = prog.withPos( eta_scaled + a2_s );
prog = prog.withPos( eta_scaled - b2_s );  prog = prog.withPos( eta_scaled + b2_s );

% Objective: weighted data 1-norm + (optional) physics tether + (optional) L1 on coefs
% Per-sample weights expand to residual entries (x,y)
W = repelem(opts.sample_weights, 2);      % length 2N
obj = 1e3 * sum(W .* delta);

if opts.lambda_dx_y     > 0, obj = obj + opts.lambda_dx_y     * sum(delta_dx); end
if opts.lambda_coeff_L1 > 0
    w = opts.l1_weights;  % P x 1
    obj = obj + opts.lambda_coeff_L1 * ( w'*(ua1+ub1+ua2+ub2) );
end
if use_jac
    obj = obj + opts.lambda_jac * sum(J_slacks);
end
if opts.lambda_colloc  > 0 && ~isempty(colloc_terms)
    obj = obj + opts.lambda_colloc * sum(colloc_terms);
end

options = spot_sdp_default_options();
options.verbose = opts.verbose;
sol = prog.minimize(obj, @spot_mosek, options);

% Recover *unscaled* coefficients for output (compatible with your pipeline)
a1 = (double(sol.eval(a1_s)) ./ scale);
b1 = (double(sol.eval(b1_s)) ./ scale);
a2 = (double(sol.eval(a2_s)) ./ scale);
b2 = (double(sol.eval(b2_s)) ./ scale);

vars.a1 = a1; vars.b1 = b1; vars.a2 = a2; vars.b2 = b2;
vars.delta = delta;
vars.scale = scale;
vars.info  = struct('used_jac',use_jac,'used_colloc',opts.lambda_colloc>0);
end
