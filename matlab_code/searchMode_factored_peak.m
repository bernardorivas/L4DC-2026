function [sol, prog, cost_vec, vars] = searchMode_factored_peak(dz, dz_tilde, prog_base, q_hint, rho)
% searchMode_factored_peak.m
% ------------------------------------------------------------
% Factored 4-mode assignment with margin sharpening (LP).
% λ_i = [(1-u_i)(1-v_i), u_i(1-v_i), (1-u_i)v_i, u_i v_i]
%     = [1 - u_i - v_i + w_i,  u_i - w_i,  v_i - w_i,  w_i],
% with McCormick envelopes for w_i ≈ u_i v_i.
%
% Adds winner-gap variables s_i with linear constraints:
%    s_i <= λ_{i,q*} - λ_{i,r}  for all r≠q*,
% and objective:  minimize  Σ||r_i||_1 * 1e3  -  ρ Σ s_i.
%
% Inputs:
%   dz          : N×n measured velocities
%   dz_tilde{q} : 1×4 cell, each N×n predicted vel under mode q
%   prog_base   : spotsosprog
%   q_hint      : N×1 ints in {1..4} (winner hint per sample)
%   rho         : scalar >= 0 (margin reward weight, e.g., 1e-3)
%
% Outputs:
%   sol, prog, cost_vec
%   vars.u, vars.v, vars.w   : N×1 msspoly
%   vars.lam                 : N×4 msspoly
%   vars.t                   : N*n×1 msspoly (L1 slacks)
%   vars.s                   : N×1 msspoly (margins)

    % ----- basic sizes / checks -----
    N = size(dz, 1);
    n = size(dz, 2);
    M = numel(dz_tilde);
    if M ~= 4
        error('searchMode_factored_peak expects exactly 4 modes.');
    end
    if nargin < 5, rho = 0; end
    if ~isequal(size(q_hint), [N,1])
        error('q_hint must be N×1 integers in {1..4}.');
    end
    for q = 1:M
        if ~isequal(size(dz_tilde{q}), [N, n])
            error('dz_tilde{%d} must be N×n.', q);
        end
    end

    prog = prog_base;

    % ----- latent variables u, v, w in [0,1] -----
    [prog, u] = prog.newPos(N);
    [prog, v] = prog.newPos(N);
    [prog, w] = prog.newPos(N);
    prog = prog.withPos(1 - u);
    prog = prog.withPos(1 - v);
    prog = prog.withPos(1 - w);

    % McCormick envelopes for w ≈ u*v:
    %   w <= u,  w <= v,  w >= u + v - 1
    prog = prog.withPos(u - w);
    prog = prog.withPos(v - w);
    prog = prog.withPos(w - (u + v - 1));

    % ----- affine λ from (u,v,w) -----
    lam = msspoly(zeros(N,4));
    lam(:,1) = 1 - u - v + w;  % (1-u)(1-v)
    lam(:,2) = u - w;          % u(1-v)
    lam(:,3) = v - w;          % (1-u)v
    lam(:,4) = w;              % uv

    % Ensure nonnegativity of λ components (sum-to-one holds automatically)
    prog = prog.withPos(lam(:));

    % ----- residuals and ℓ1 epigraph -----
    cost_vec = msspoly([]);
    for i = 1:N
        ri = dz(i,:).';
        for q = 1:4
            ri = ri - lam(i,q) * dz_tilde{q}(i,:).';
        end
        cost_vec = [cost_vec; ri]; %#ok<AGROW>
    end

    [prog, t] = prog.newPos(length(cost_vec));
    prog = prog.withPos(t - cost_vec);
    prog = prog.withPos(t + cost_vec);

    % ----- margin sharpening: s_i ≤ λ_{i,q*} - λ_{i,r} -----
    [prog, s] = prog.newPos(N);
    for i = 1:N
        qi = q_hint(i);
        if qi < 1 || qi > 4
            error('q_hint(%d) = %d is out of range {1..4}.', i, qi);
        end
        for r = 1:4
            if r == qi, continue; end
            % lam(i,qi) - lam(i,r) - s(i) >= 0
            prog = prog.withPos(lam(i,qi) - lam(i,r) - s(i));
        end
    end

    % ----- solve LP -----
    options = spot_sdp_default_options();
    options.verbose = 12;
    obj = sum(t) * 1e3 - rho * sum(s);
    sol = prog.minimize(obj, @spot_mosek, options);

    % ----- pack outputs -----
    vars.u   = u;
    vars.v   = v;
    vars.w   = w;
    vars.lam = lam;
    vars.t   = t;
    vars.s   = s;
end
