function [sol, prog, cost_vec, vars] = searchNetwork(xop, dz, lambda_used, prog_base, C, alpha, beta)
%% searchNetwork: identify adjacency matrices a_j and node dynamics alpha_f
% xop: N x n state data
% dz: N x n derivative data
% lambda_used: N x Q mode assignments (each row sums to 1)
% alpha_f: 3 x 1 current guess of [alpha_0; alpha_1; alpha_2]

N = size(xop,1);
n = size(xop,2);
Q = size(lambda_used,2);

eta = 5;           % upper bound for alpha_f entries

[prog, a] = prog_base.newFree(Q * n^2); % a = [a1; a2] flattened adjacency matrix entries
for j = 1:Q
    a_j = a((j-1)*n^2 + 1 : j*n^2);
    
    % Set diagonal entries (self-loops) to 0
    diag_idx = 1:n+1:n^2; % diagonal indices of nxn matrix flattened
    for d = diag_idx
        prog = prog.withEqs(a_j(d));
    end
end


[prog, alpha_f] = prog.newFree(3);        % scalar node dynamics parameters

% bounds on alpha and a
prog = prog.withPos(eta - alpha_f);
prog = prog.withPos(eta + alpha_f);
prog = prog.withPos(a);
prog = prog.withPos(1 - a);

% Slack variables for 1-norm loss
[prog, delta] = prog.newPos(N * n);
cost_vec = delta;

for i = 1:N
    xi = xop(i,:)';
    dxi = dz(i,:)';

    % compute f(xi)
    fx = alpha_f(1) + alpha_f(2) * xi + alpha_f(3) * (xi.^2);

    % build L_j * x for each j using current a_j
    Lx_total = zeros(n,1);
    for j = 1:Q
        a_j = a((j-1)*n^2 + 1 : j*n^2);
        A_j = reshape(a_j, n, n);
        D_j = diag(sum(A_j, 2));
        L_j = alpha*D_j + beta*A_j; % Laplacian with alpha = 1, beta = -1
        Lx_total = Lx_total + lambda_used(i,j) * L_j * xi;
    end

    pred = fx + C*Lx_total;
    for k = 1:n
        res = dxi(k) - pred(k);
        prog = prog.withPos(delta((i-1)*n + k) - res);
        prog = prog.withPos(delta((i-1)*n + k) + res);
    end
end

options = spot_sdp_default_options();
options.verbose = 12;
[sol, prog] = prog.minimize(sum(delta) * 1e3, @spot_mosek, options);

vars.a1 = a(1:n^2);
vars.a2 = a(n^2+1:2*n^2);
% vars.a3 = a(... )
vars.alpha_f = alpha_f;
vars.delta = delta;
end
