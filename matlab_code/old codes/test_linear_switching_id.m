%% bilevel_switching_linear_id_monomial.m
% Identify A1, A2 and mode assignments in a switching linear system
% using 1-norm + monomial basis

clear; clc; close all;

%% 1) CVX setup
cvx_clear;
cvx_setup;
cvx_solver mosek

%% 2) Load data
load("switching_ho_data.mat") % contains data.y, data.dy, data.mode
X_full    = data.y;              % N_full × 2
Xdot_full = data.dy;             % N_full × 2
mode_full = data.mode;           % N_full × 1
N_full    = size(X_full,1);

%% 3) Randomly subsample M rows once and for all
M = min(2000,N_full);             % pick 2000 or fewer if data smaller
id = randperm(N_full,M);
X    = X_full(id,:);             % M × 2
Xdot = Xdot_full(id,:);          % M × 2
mode = mode_full(id);            % M × 1
N    = M;                        % we now work with N=M points

n    = 2;
kappa = 1;           % degree of monomials

%% 4) Build monomial basis v(x) = [1; x1; x2]
kappa = 1;                       % total degree
x  = msspoly('x',2);
vx = monomials(x,0:kappa);       % {1, x1, x2}
vlen = length(vx);               % should be 3

%% 5) Precompute the N×vlen "lifted" V-matrix
V = zeros(N,vlen);
for i=1:N
    V(i,:) = double( subs(vx, x, X(i,:)') )';
end

%% 6) Initialize lambda randomly (rows sum to 1)
lambda = rand(N,2);
lambda = lambda./sum(lambda,2);

max_iter = 100;
tol      = 1e-6;
cost_log = zeros(max_iter,1);

%% 7) Alternating optimization
for iter = 1:max_iter

    % —————————————————————
    % a) LP step: fix lambda, solve for A1(x), A2(x)
    % —————————————————————
    cvx_begin quiet
        variables a1(4, vlen) a2(4, vlen)
        % We'll collect the residuals in an N×2 matrix:
        expression total_res(N,2)

        for i = 1:N
            vxi = V(i, :);             % 1×vlen
            % reconstruct vec(A1(x_i)) and vec(A2(x_i))
            A1vec = a1 * vxi';         % 4×1
            A2vec = a2 * vxi';         % 4×1

            A1x   = reshape(A1vec, 2, 2);   % 2×2
            A2x   = reshape(A2vec, 2, 2);

            % predicted dx for each mode
            f1    = A1x * X(i, :)';
            f2    = A2x * X(i, :)';

            % convex combination
            fx    = lambda(i,1)*f1 + lambda(i,2)*f2;

            total_res(i,:) = (Xdot(i,:)' - fx)';
        end

        minimize( norm(total_res, 1) )
    cvx_end
    
    % log CVX status
    fprintf('Iter %3d: CVXstatus=%s, CVXobj=%.4e\n', ...
            iter, cvx_status, cvx_optval );
    % —————————————————————
    % b) Re-assign lambda via 1-norm residuals
    % —————————————————————
    res_mat = zeros(N,2);
    for i = 1:N
        vxi = V(i, :);

        A1vec = a1 * vxi';
        A2vec = a2 * vxi';

        A1x   = reshape(A1vec, 2, 2);
        A2x   = reshape(A2vec, 2, 2);

        e1    = norm( Xdot(i,:)' - A1x*X(i,:)' , 1);
        e2    = norm( Xdot(i,:)' - A2x*X(i,:)' , 1);

        res_mat(i,:) = [e1, e2];
    end

    inv_res = 1 ./ (res_mat + 1e-6);           % avoid div by zero
    lambda  = inv_res ./ sum(inv_res, 2);     % re-normalize rows

    cost_log(end+1) = sum( min(res_mat,[],2) );

    % compute "average confidence" in each mode 
    mode_confidence(iter,:) = mean(lambda,1);
    
    % print cost & confidence
    fprintf('         cost=%.4e   avg λ=[%.3f  %.3f]\n', ...
            cost_log(iter), mode_confidence(iter,1), mode_confidence(iter,2) );
    
    if iter>1 && abs(cost_log(end)-cost_log(end-1))<tol
        fprintf('Converged at iteration %d (cost = %.4e)\n', iter, cost_log(end));
        break
    end
end

%% 8) Recover mode assign estimate and plot
[~,mode_est] = max(lambda,[],2);

figure;
stairs(1:N,mode_est,'LineWidth',1.5); hold on
stairs(1:N,mode,'--k','LineWidth',1);
xlabel('sample #'); ylabel('mode');
legend('estimated','true','Location','best');
title('Mode assignments on subsampled data');

figure;
semilogy(cost_log,'-o','LineWidth',1.2); grid on
xlabel('iteration'); ylabel('total 1-norm residual');
title('convergence (subsampled N=2000)');

