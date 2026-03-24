function [sol, prog, cost_vec, vars] = searchMode(dz, dz_tilde, prog_base)
%%
% dz: N x n, N measurements of nx1 vector (n: dim of system)
% dz_tilde{q}: N x n predicted dot{x} trajectory from mode q (Aq*xi)
% |dx - \sum \lam_i * dx_tilde_i|_1
%%
N = size(dz, 1); % # of measurements
M = length(dz_tilde); % # of modes
[prog, lam] = prog_base.newPos(N * M);
lam = reshape(lam, [N, M]); % NxM matrix of predicted lambdas

cost_vec = [];
parfor k = 1:N
    dz_tilde_k = 0;
    for q = 1:M
        dz_tilde_k = dz_tilde_k + dz_tilde{q}(k, :)' * lam(k, q); % e.g., dz_tilde{1}(k, :)' = A1*xk
    end
    cost_vec = [cost_vec; dz(k, :)' - dz_tilde_k]; % cost_vec is a nNx1 column vector 
end
[prog, t] = prog.newPos(length(cost_vec)); % declare nNx1 positive slack variables vector t
prog = prog.withPos(t - cost_vec); % x - x_tilde <= t
prog = prog.withPos(t + cost_vec); % -t <= x - x_tilde

prog = prog.withPos(1 - lam(:)); % lam <= 1
prog = prog.withPos(lam(:));     % lam >= 0

for k = 1:N
    prog = prog.withEqs(1 - sum(lam(k, :))); % \sum_{q=1}^M lam_{k,q} = 1
end

options = spot_sdp_default_options();
options.verbose = 12;
sol = prog.minimize(sum(t) * 1e3, @spot_mosek, options);

vars.lam = lam;
vars.t   = t;
end