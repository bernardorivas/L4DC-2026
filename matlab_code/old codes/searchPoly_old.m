function [sol, prog, cost_vec, vars] = searchPoly_old(X, Xdot, lambda, prog_base, phi_d, opts)
% Minimal searchPoly.m (crude baseline)
% - 2-mode polynomial ID via L1 fit of residuals only.
% - No column scaling, no coefficient L1, no Jacobian, no collocation, no tethers.
% - Keeps interface/outputs compatible with your pipeline.
%
% Inputs:
%   X       : N x 2 states [x, y]          (unused here except for size consis.)
%   Xdot    : N x 2 velocities [xdot, ydot]
%   lambda  : N x 2 soft/hard mode weights (rows sum to 1)
%   prog_base : Spotless base program
%   phi_d   : N x P monomial basis at X
%   opts    : (optional) struct
%       .verbose (default 12)  MOSEK verbosity
%
% Outputs:
%   sol, prog, cost_vec, vars  (vars has a1,b1,a2,b2 and delta)

if nargin < 6, opts = struct; end
if ~isfield(opts,'verbose'), opts.verbose = 12; end

[N, P] = size(phi_d);
prog = prog_base;

% Decision variables: coefficients in the (unscaled) basis
[prog, a1] = prog.newFree(P);  % coef for xdot in mode 1
[prog, b1] = prog.newFree(P);  % coef for ydot in mode 1
[prog, a2] = prog.newFree(P);  % coef for xdot in mode 2
[prog, b2] = prog.newFree(P);  % coef for ydot in mode 2

% Build residuals r_i = Xdot(i,:)' - [lam1 f1(phi_i) + lam2 f2(phi_i)]
cost_vec = [];   % stack as [r_x(1); r_y(1); r_x(2); r_y(2); ...]
for i = 1:N
    phi_i = phi_d(i,:).';         % P x 1 numeric
    lam1  = lambda(i,1);
    lam2  = lambda(i,2);

    f1 = [a1' * phi_i; b1' * phi_i];  % mode 1 prediction
    f2 = [a2' * phi_i; b2' * phi_i];  % mode 2 prediction
    fx = lam1 * f1 + lam2 * f2;

    residual = Xdot(i,:)' - fx;       % 2x1 msspoly
    cost_vec = [cost_vec; residual];
end

% L1 slacks for |residuals|
[prog, delta] = prog.newPos(length(cost_vec));
prog = prog.withPos(delta - cost_vec);
prog = prog.withPos(delta + cost_vec);

% Objective: minimize sum |residuals|
obj = sum(delta);

% Solve
options = spot_sdp_default_options();
options.verbose = opts.verbose;
[sol, prog] = prog.minimize(obj, @spot_mosek, options);

% Extract numeric coefficients
vars.a1    = double(sol.eval(a1));
vars.b1    = double(sol.eval(b1));
vars.a2    = double(sol.eval(a2));
vars.b2    = double(sol.eval(b2));
vars.delta = delta;   % keep symbolic handle for downstream inspection
vars.scale = ones(P,1); % for compatibility with newer codepaths
vars.info  = struct('used_jac',false,'used_colloc',false);

end
