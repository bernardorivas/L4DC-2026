function [sol, prog, cost_vec, vars] = searchMode_moment(dz, dz_tilde, prog_base)
%% Moment/PSD (Shor) relaxation for mode assignment
% dz         : N x n   measured velocities
% dz_tilde{q}: N x n   predicted velocities under mode q (e.g., A_q * x)
% Objective  : min sum_i || dz(i,:) - sum_q lambda_iq * dz_tilde{q}(i,:) ||_1
% Constraints: lambda^i in simplex via order-1 moment relaxation:
%              [1 lambda^i'; lambda^i Lambda^i] >= 0,
%              diag(Lambda^i) = lambda^i, sum(lambda^i)=1, 0<=lambda^i<=1.

N = size(dz, 1);
M = numel(dz_tilde);
n = size(dz, 2);

prog = prog_base;

% First-order moments (lambda) we will fill per-sample to keep Mi block-diagonal
lam_cell = cell(N,1);
Lam_cell = cell(N,1);     % second-moment surrogate (symmetric)
Mi_cell  = cell(N,1);     % moment matrices for PSD

% Build linear residuals and slack t for L1 norm
cost_vec = msspoly([]);

% Per-sample construction
for i = 1:N
    % lambda^i: first-order moments
    [prog, lam_i] = prog.newFree(M);            % free first; we’ll box with 0<=lam<=1
    lam_cell{i} = lam_i;
    
    % Lambda^i: symmetric second-moment surrogate
    [prog, L_i] = prog.newSym(M);
    Lam_cell{i} = L_i;

    % Moment matrix M_i >= 0
    Mi = [1,         lam_i.';
          lam_i,     L_i      ];
    Mi_cell{i} = Mi;
    prog = prog.withPSD(Mi);

    % Localizing equalities for lambda_j^2 - lambda_j = 0 at r=1:
    % enforce diag(L_i) == lam_i
    prog = prog.withEqs(diag(L_i) - lam_i);

    % Simplex constraints 0 <= lam_i <= 1, sum(lam_i) = 1
    prog = prog.withPos(lam_i);
    prog = prog.withPos(1 - lam_i);
    prog = prog.withEqs(sum(lam_i) - 1);

    % Residual for sample i: dz(i,:)' - sum_q lam_i(q) * dz_tilde{q}(i,:)'
    r_i = dz(i,:).';
    for q = 1:M
        % dz_tilde{q}(i,:)' times lam_i(q)
        r_i = r_i - lam_i(q) * dz_tilde{q}(i,:).';
    end
    cost_vec = [cost_vec; r_i]; %#ok<AGROW>
end

% L1 slack for all residual components
[prog, t] = prog.newPos(length(cost_vec));
prog = prog.withPos(t - cost_vec);
prog = prog.withPos(t + cost_vec);

% Solve
options = spot_sdp_default_options();
options.verbose = 12;
[sol, prog] = prog.minimize(sum(t) * 1e3, @spot_mosek, options);

% Package outputs
% Stack lambdas into an N x M matrix for convenient downstream use
lam_mat = msspoly(zeros(N, M));
for i = 1:N
    lam_mat(i,:) = lam_cell{i}.';
end

vars.lam      = lam_mat;    % first-order moments y_{e_j} = lambda_j
vars.t        = t;
vars.Mi       = Mi_cell;    % moment matrices, for rank/tightness checks
vars.Lambda   = Lam_cell;   % second-moment blocks (optional diagnostics)
end
