function [sol, prog, cost_vec, vars] = searchMatrixAffine(X, Xdot, lambda, prog_base)
% Identify {A_q, b_q} with 1-norm loss from fixed λ
% X, Xdot : N×n
% lambda  : N×Q   (rows in simplex; hard or soft OK)
% prog_base: spotsosprog
%
% Model:  xdot_i ≈ sum_q λ_{iq} (A_q x_i + b_q)

[N, n] = size(X);
Q = size(lambda,2);
prog = prog_base;

% decision variables per mode
vars.a = cell(Q,1);   % vec(A_q) length n^2
vars.b = cell(Q,1);   % b_q     length n
eta = 10;             % simple box (optional)

for q = 1:Q
    [prog, vars.a{q}] = prog.newFree(n^2);
    [prog, vars.b{q}] = prog.newFree(n);
    % optional bounds (helps conditioning)
    prog = prog.withPos(eta - vars.a{q}); prog = prog.withPos(vars.a{q} + eta);
    prog = prog.withPos(eta - vars.b{q}); prog = prog.withPos(vars.b{q} + eta);
end

% build residuals
cost_vec = msspoly([]);   % collect all components (N*n×1)
for i = 1:N
    xi  = X(i,:)';
    dxi = Xdot(i,:)';
    fx  = 0*dxi;
    for q = 1:Q
        Aq_xi = reshape(vars.a{q}, n, n) * xi;
        bq    = vars.b{q};
        fx    = fx + lambda(i,q) * (Aq_xi + bq);
    end
    cost_vec = [cost_vec; dxi - fx]; %#ok<AGROW>
end

% ℓ1 slack
[prog, delta] = prog.newPos(length(cost_vec));
prog = prog.withPos(delta - cost_vec);
prog = prog.withPos(delta + cost_vec);

% solve
options = spot_sdp_default_options(); options.verbose = 12;
sol = prog.minimize(sum(delta) * 1e3, @spot_mosek, options);

vars.delta = delta;
end
