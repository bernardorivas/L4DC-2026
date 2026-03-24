function [sol, prog, cost_vec, vars] = searchMode_factored(dz, dz_tilde, prog_base)
% searchMode_factored.m
% ------------------------------------------------------------
% Factored mode assignment for 4-mode systems via two latent
% binaries u (horizontal) and v (vertical) with McCormick envelopes.
%
% Inputs:
%   dz          : N×n measured velocities
%   dz_tilde{q} : 1×4 cell, each N×n predicted vel under mode q
%                  (dy_q = A_q * x + b_q evaluated on the batch)
%   prog_base   : a spotsosprog instance
%
% Model:
%   lambda_i = [ (1-u_i)(1-v_i), u_i(1-v_i), (1-u_i)v_i, u_i v_i ]
%            = [ 1 - u_i - v_i + w_i,  u_i - w_i,  v_i - w_i,  w_i ],
%   with 0≤u_i,v_i,w_i≤1 and McCormick:
%          w_i ≤ u_i,  w_i ≤ v_i,  w_i ≥ u_i + v_i - 1.
%
% Objective:
%   min sum_i || dz(i,:) - sum_q lambda_{iq} * dz_tilde{q}(i,:) ||_1
%
% Output:
%   sol        : Spotless solution
%   prog       : final program
%   cost_vec   : stacked residual components (N*n×1 msspoly)
%   vars       : struct with fields
%                  .u, .v, .w     (N×1 msspoly)
%                  .lam           (N×4 msspoly)
%                  .t             (N*n×1 msspoly)  L1 slacks
%
% Notes:
%   - Keeps the whole subproblem an LP (all linear constraints).
%   - Designed for Q=4; throws error otherwise.

    N = size(dz, 1);
    n = size(dz, 2);
    M = numel(dz_tilde);
    if M ~= 4
        error('searchMode_factored: expects exactly 4 modes (got M=%d).', M);
    end

    % Basic checks on dimensions
    for q = 1:M
        if ~isequal(size(dz_tilde{q}), [N, n])
            error('dz_tilde{%d} must be N×n (got %dx%d).', q, size(dz_tilde{q},1), size(dz_tilde{q},2));
        end
    end

    prog = prog_base;

    % ----- Variables: u, v, w in [0,1] (N×1 each) -----
    [prog, u] = prog.newPos(N);
    [prog, v] = prog.newPos(N);
    [prog, w] = prog.newPos(N);

    % Box: u,v,w <= 1
    prog = prog.withPos(1 - u);
    prog = prog.withPos(1 - v);
    prog = prog.withPos(1 - w);

    % McCormick envelopes for w ≈ u*v
    %   w ≤ u,  w ≤ v,  w ≥ u+v-1
    prog = prog.withPos(u - w);                 % u - w >= 0  (i.e., w <= u)
    prog = prog.withPos(v - w);                 % v - w >= 0  (i.e., w <= v)
    prog = prog.withPos(w - (u + v - 1));       % w - u - v + 1 >= 0

    % ----- Build lambda_i (N×4), each entry affine in (u,v,w) -----
    % lam(:,1) = 1 - u - v + w
    % lam(:,2) = u - w
    % lam(:,3) = v - w
    % lam(:,4) = w
    lam = msspoly(zeros(N,4));
    lam(:,1) = 1 - u - v + w;
    lam(:,2) = u - w;
    lam(:,3) = v - w;
    lam(:,4) = w;

    % (Optional but harmless) nonnegativity of each lambda component
    prog = prog.withPos(lam(:));

    % Sum-to-one holds automatically:
    % (1-u-v+w) + (u-w) + (v-w) + w = 1

    % ----- Residuals and L1 slack -----
    % r_i = dz(i,:)' - sum_q lam(i,q) * dz_tilde{q}(i,:)'  ∈ R^n
    cost_vec = msspoly([]);
    for i = 1:N
        ri = dz(i,:).';
        for q = 1:4
            ri = ri - lam(i,q) * dz_tilde{q}(i,:).';
        end
        cost_vec = [cost_vec; ri]; %#ok<AGROW>
    end

    % L1 epigraph: -t <= r <= t, t >= 0
    [prog, t] = prog.newPos(length(cost_vec));
    prog = prog.withPos(t - cost_vec);
    prog = prog.withPos(t + cost_vec);

    % ----- Solve LP -----
    options = spot_sdp_default_options();
    options.verbose = 12;
    % keep same scaling as your other routines (×1e3)
    sol = prog.minimize(sum(t) * 1e3, @spot_mosek, options);

    % ----- Pack outputs -----
    vars.u   = u;
    vars.v   = v;
    vars.w   = w;
    vars.lam = lam;
    vars.t   = t;
end
