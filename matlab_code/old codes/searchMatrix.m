function [sol, prog, cost_vec, vars] = searchMatrix(X, Xdot, lambda, prog_base)
% Identify A1, A2 (linear dynamics matrices) using 1-norm regression
% Input:
%   - X:     N×n data
%   - Xdot:  N×n velocity vectors
%   - lambda: N×2 mode assignment weights
%   - prog_base: a spotless program
% Output:
%   - sol: Spotless solution
%   - prog: final program
%   - cost_vec: residuals
%   - vars: struct containing decision variables (a1, a2, delta)

[N, n] = size(X); % N: # of measurements, n: dim of system (n=2 for harmonic oscillator)
prog = prog_base;

% declare decision variables: A1 and A2 are nxn matrices (vectoried as n^2x1)
[prog, a1] = prog.newFree(n^2);
[prog, a2] = prog.newFree(n^2);

cost_vec = [];
parfor i = 1:N
    xi = X(i, :)'; % ith measurement as column vector
    dxi = Xdot(i, :)'; % ith measurement of velocity as column vector

    % construct predicted vector field fx
    A1xi = reshape(a1,2,2) * xi;
    A2xi = reshape(a2,2,2) * xi;
    fx = lambda(i,1) * A1xi + lambda(i,2) * A2xi;

    cost_vec = [cost_vec; dxi - fx];
end

% introduce slack variables for 1-norm loss
[prog, delta] = prog.newPos(length(cost_vec));
prog = prog.withPos(delta - cost_vec);
prog = prog.withPos(delta + cost_vec);

% optilnal: box constraints on A1, A2 entries
eta = 10;  % Adjust if needed
prog = prog.withPos(eta - a1); prog = prog.withPos(a1 + eta); % -η ≤ a1k ≤ η
prog = prog.withPos(eta - a2); prog = prog.withPos(a2 + eta); % -η ≤ a2k ≤ η

options = spot_sdp_default_options();
options.verbose = 12;
[sol, prog] = prog.minimize(sum(delta) * 1e3, @spot_mosek, options);

% Output variables
vars.a1 = a1;
vars.a2 = a2;
vars.delta = delta;

end