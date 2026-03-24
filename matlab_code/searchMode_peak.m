function [sol, prog, cost_vec, vars] = searchMode_peak(dz, dz_tilde, prog_base, q_hint, rho)
%% LP with margin reward to sharpen λ toward a hinted winner per sample
% dz:        N×n measured velocities
% dz_tilde:  1×M cell, each N×n predicted vel under mode q
% q_hint:    N×1 integers in {1..M} (winner hint per sample)
% rho:       scalar >= 0 (margin reward weight)

N = size(dz,1);
M = numel(dz_tilde);
n = size(dz,2);

prog = prog_base;

% λ variables (N×M), simplex per row
[prog, lam] = prog.newPos(N*M);
lam = reshape(lam, [N, M]);

% residual vector for L1
cost_vec = msspoly([]);

% build cost residuals
for k = 1:N
    r_k = dz(k,:).';
    for q = 1:M
        r_k = r_k - lam(k,q) * dz_tilde{q}(k,:).';
    end
    cost_vec = [cost_vec; r_k]; %#ok<AGROW>
end

% L1 slack
[prog, t] = prog.newPos(length(cost_vec));
prog = prog.withPos(t - cost_vec);
prog = prog.withPos(t + cost_vec);

% simplex constraints per sample
for k = 1:N
    prog = prog.withEqs(1 - sum(lam(k,:)));  % sum_q λ_{kq} = 1
end
prog = prog.withPos(1 - lam(:));             % λ <= 1

% ---- margin variables s_i and linear gap constraints ----
[prog, s] = prog.newPos(N);                  % s >= 0
for k = 1:N
    qstar = q_hint(k);
    for r = 1:M
        if r == qstar, continue; end
        % s_k <= λ_{k,q*} - λ_{k,r}  ==>  s_k - λ_{k,q*} + λ_{k,r} <= 0
        prog = prog.withPos( 0 - ( s(k) - lam(k,qstar) + lam(k,r) ) ); % linear ineq
    end
end

% solve: minimize sum|residuals| - rho * sum s
options = spot_sdp_default_options(); options.verbose = 12;
obj = sum(t) * 1e3 - rho * sum(s);  % keep your 1e3 scaling
sol = prog.minimize(obj, @spot_mosek, options);

vars.lam = lam; vars.t = t; vars.s = s;
end
