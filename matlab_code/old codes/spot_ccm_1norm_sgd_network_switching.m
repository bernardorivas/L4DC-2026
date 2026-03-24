%% spot_ccm_1norm_sgd_network_switching.m
% Bilevel convex optimization for identifying switching network dynamics

clear; clc; close all;

% Load data
load("switching_network_data.mat");
X = data.x;      % N x 3
Xdot = data.dx;  % N x 3
mode_true = data.mode;

% Parameters
n = 3;          % number of agents (nodes)
Q = 2;          % number of modes
N = size(X,1);  % number of samples

% Coupling parameters
alpha = 1.0; beta = -1.0;
C = -1.0;  % scalar coupling channel matrix

A1_true = [0 1 1; 1 0 1; 1 1 0];
A2_true = [0 1 0; 0 0 1; 1 0 0];

% Initialization
nIter = 10;
idx_batch = randperm(N, 2000);
xop = X(idx_batch,:);
dz = Xdot(idx_batch,:);
true_mode = mode_true(idx_batch,:);

% Initialize soft labels and node dynamics
lambda_soft = rand(size(xop,1), Q);
lambda_soft = lambda_soft ./ sum(lambda_soft,2);
alpha_f = randn(3,1);  % coefficients of f(x) = a0 + a1 x + a2 x^2
a1 = rand(9,1);  % vectorized A1 (3x3)
a2 = rand(9,1);  % vectorized A2
A1_init = reshape(double(a1 > 0.5), [3,3]);
A2_init = reshape(double(a2 > 0.5), [3,3]);
A1_err0 = A1_init - A1_true;
A2_err0 = A2_init - A2_true;
adj_err0 = sum(abs(A1_err0(:))) + sum(abs(A2_err0(:)));

cost_log = zeros(nIter, 1);
adj_err = zeros(nIter, 1);
mode_err = zeros(nIter, 1);

adj_err(1) = adj_err0;  % store initial mismatch

for iter = 1:nIter
    %% === Step-1 : forward simulation with current parameters ===
    % Compute Laplacian matrices from a1 and a2
    A1 = reshape(a1, [3,3]); D1 = diag(sum(A1,2)); L1 = alpha*D1 + beta*A1;
    A2 = reshape(a2, [3,3]); D2 = diag(sum(A2,2)); L2 = alpha*D2 + beta*A2;

    % Evaluate dy_tilde for each mode
    dy_tilde = cell(1,Q);
    for i = 1:size(xop,1)
        xi = xop(i,:)';
        fxi = alpha_f(1) + alpha_f(2)*xi + alpha_f(3)*(xi.^2);
        dy_tilde{1}(i,:) = (fxi + C * L1 * xi)';
        dy_tilde{2}(i,:) = (fxi + C * L2 * xi)';
    end

    %% === Step-2 : Mode Assignment ===
    prog = spotsosprog;
    [sol_mode, prog, ~, vars_mode] = searchMode(dz, dy_tilde, prog);
    lambda_soft = double(sol_mode.eval(vars_mode.lam));
    [~, mode_op] = max(lambda_soft, [], 2);

    % Resolve label permutation
    if ~isempty(true_mode)
        mismatch1 = nnz(mode_op ~= true_mode);
        mismatch2 = nnz((3 - mode_op) ~= true_mode);
        if mismatch2 < mismatch1
            mode_op = 3 - mode_op;
            lambda_soft = lambda_soft(:, [2 1]);
        end
        mode_err(iter) = nnz(mode_op ~= true_mode);
    end

    %% === Step-3 : Network structure and f(x) identification ===
    if iter <= 2
        lambda_used = lambda_soft;
    else
        lambda_used = full(sparse(1:length(mode_op), mode_op, 1, length(mode_op), Q));
    end

    prog2 = spotsosprog;
    [sol_net, prog2, cost_vec, vars_net] = searchNetwork(xop, dz, lambda_used, prog2, C, alpha, beta);

    % Extract identified parameters
    alpha_f = double(sol_net.eval(vars_net.alpha_f));
    a1 = double(sol_net.eval(vars_net.a1));
    a2 = double(sol_net.eval(vars_net.a2));

    % Threshold the adjacency matrix entries inside the loop
    if iter >= 0
        a1 = double(a1 > 0.5);
        a2 = double(a2 > 0.5);
    end

    A1_id = reshape(a1 > 0.5, [3,3]);
    A2_id = reshape(a2 > 0.5, [3,3]);
    
    A1_err = A1_id - A1_true;
    A2_err = A2_id - A2_true;
    % If ground truth adjacency matrices are available:
    adj_err(iter+1) = sum(abs(A1_err(:))) + sum(abs(A2_err(:)));

    % Log cost
    cost_log(iter) = double(sol_net.eval(sum(vars_net.delta(:))));
    fprintf("Iter %2d | Loss: %.4e | Mode mismatch: %d | Adj Mismatch: %.2e\n", ...
        iter, cost_log(iter), mode_err(iter), adj_err(iter));
end

%% Compare Adjacency Matrices
% If ground truth adjacency matrices are available:
fprintf("Adjacency MSE: %.3e, %.3e\n", norm(A1_id - A1_true), norm(A2_id - A2_true));

%% Plotting results
figure(1); clf;
semilogy(abs(cost_log), '-o');
title('Network Dynamics Identification Error', 'Interpreter','latex'); xlabel('Iteration', 'Interpreter','latex'); ylabel('Cost', 'Interpreter','latex'); grid on;

figure(2); clf;
plot(mode_err, '-s');
title('Mode Classification Error', 'Interpreter','latex'); xlabel('Iteration', 'Interpreter','latex'); ylabel('Mismatch', 'Interpreter','latex'); grid on;

figure(3); clf;
plot(0:nIter, adj_err, '-.o');
title('Adjacency Matrices Identification Error', 'Interpreter', 'latex'); xlabel('Iteration', 'Interpreter','latex'); ylabel('Mismatch', 'Interpreter','latex'); grid on;

%% Scatter Plot

% Assuming mode_op ∈ {1, 2}
idx_mode1 = (mode_op == 1);
idx_mode2 = (mode_op == 2);

figure(4); clf; hold on; grid on; axis equal;

% Choose colors (adjust as you prefer)
color1 = [0.2, 0.6, 1.0];  % light blue for Mode 1
color2 = [1.0, 0.5, 0.1];  % reddish-orange for Mode 2

% Scatter 3D points by mode
h1 = scatter3(xop(idx_mode1,1), xop(idx_mode1,2), xop(idx_mode1,3), ...
              10, color1, 'filled', 'DisplayName', 'Mode 1');
h2 = scatter3(xop(idx_mode2,1), xop(idx_mode2,2), xop(idx_mode2,3), ...
              10, color2, 'filled', 'DisplayName', 'Mode 2');

legend([h1 h2], 'Location', 'best');
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
zlabel('$x_3$', 'Interpreter', 'latex');
title('Mode Identification in State Space', 'Interpreter', 'latex');
view(20, 30);

%% Switching Surface Recovery
kappa = 2;   % degree of switching surface polynomial
eta = 10;
epsilon = 1e-2;
beta = 1e-2;

% Construct signed labels
sigma = 2 * (mode_op == 1) - 1;

% Run convex optimization to recover f(x) ≈ x1^2 + x2^2 + x3^2 - R^2
[a_opt, z_val, diagnostics] = searchSwitchingSurface_softmargin(xop, sigma, kappa, eta, epsilon, beta);

%% Plot Identified Switching Surface
figure(5); clf; hold on; axis equal; grid on;

% Scatter by mode
h1 = scatter3(xop(idx_mode1,1), xop(idx_mode1,2), xop(idx_mode1,3), ...
              10, color1, 'filled', 'DisplayName', 'Mode 1');
h2 = scatter3(xop(idx_mode2,1), xop(idx_mode2,2), xop(idx_mode2,3), ...
              10, color2, 'filled', 'DisplayName', 'Mode 2');

% Generate grid
[x1g, x2g, x3g] = meshgrid(linspace(-5,5,40));
Xgrid = [x1g(:), x2g(:), x3g(:)];
Phi_grid = buildMonomialMatrix(Xgrid, kappa);
Zgrid = reshape(Phi_grid * a_opt, size(x1g));

% Plot isosurface of f(x) = 0
h3 = patch(isosurface(x1g, x2g, x3g, Zgrid, 0));
h3.DisplayName = '$f_{\mathrm{identified}}(x) = 0$';
isonormals(x1g, x2g, x3g, Zgrid, h3);
set(h3, 'FaceColor', [0.2,0.2,0.2], 'EdgeColor', 'none', 'FaceAlpha', 0.2);
legend([h1, h2, h3], 'Location','best', 'Interpreter','latex');
xlabel('$x_1$', 'Interpreter','latex');
ylabel('$x_2$', 'Interpreter','latex');
zlabel('$x_3$', 'Interpreter','latex');
title('Identified Switching Surface', 'Interpreter','latex');
view(20,30); camlight; lighting gouraud;

%% Plot Identified vs Ground Truth Switching Surface (3D)
figure(6); clf; hold on; axis equal; grid on;

% === Scatter points colored by mode ===
h1 = scatter3(xop(idx_mode1,1), xop(idx_mode1,2), xop(idx_mode1,3), ...
              10, color1, 'filled', 'DisplayName', 'Mode 1');
h2 = scatter3(xop(idx_mode2,1), xop(idx_mode2,2), xop(idx_mode2,3), ...
              10, color2, 'filled', 'DisplayName', 'Mode 2');

% === Grid for evaluation ===
[x1g, x2g, x3g] = meshgrid(linspace(-5,5,50));
Xgrid = [x1g(:), x2g(:), x3g(:)];

% === Identified surface: f_identified(x) = a_opt' * phi(x) ===
Phi_grid = buildMonomialMatrix(Xgrid, kappa);
Z_identified = reshape(Phi_grid * a_opt, size(x1g));

% === Ground truth surface: f_true(x) = x1^2 + x2^2 + x3^2 - delta ===
R = 3.0;
delta = R^2;
Z_true = x1g.^2 + x2g.^2 + x3g.^2 - delta;

% === Plot identified surface ===
h3 = patch(isosurface(x1g, x2g, x3g, Z_identified, 0));
isonormals(x1g, x2g, x3g, Z_identified, h3);
set(h3, 'FaceColor', [0 0 0], 'EdgeColor', 'none', 'FaceAlpha', 0.15);
h3.DisplayName = '$f_{\mathrm{identified}}(x) = 0$';

% === Plot ground truth surface ===
h4 = patch(isosurface(x1g, x2g, x3g, Z_true, 0));
isonormals(x1g, x2g, x3g, Z_true, h4);
set(h4, 'FaceColor', [0.8 0 0], 'EdgeColor', 'none', 'FaceAlpha', 0.2);
h4.DisplayName = '$f_{\mathrm{true}}(x) = 0$';

% === Labels, legend, view ===
xlabel('$x_1$', 'Interpreter','latex');
ylabel('$x_2$', 'Interpreter','latex');
zlabel('$x_3$', 'Interpreter','latex');
title('Identified vs Ground Truth Switching Surface', 'Interpreter','latex');

legend([h1 h2 h3 h4], 'Interpreter','latex', 'Location','best');
view(20, 30); camlight; lighting gouraud;
