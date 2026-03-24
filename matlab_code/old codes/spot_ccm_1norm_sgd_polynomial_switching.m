%% spot_ccm_1norm_sgd_polynomial_switching.m
% bilevel convex optimization for identifying switching polynomial vector fields

clear; clc; close all;

% Load data
load("switching_quartic_oscillator_data.mat")
X = data.y;     % N x 2
Xdot = data.dy; % N x 2
N = size(X, 1);
n = 2;          % dim of state
Q = 2;          % number of modes

% parameters
nIter = 10;                        % number of alternating iterations
d = 3;                             % degree of monomial basis
phi_d = buildMonomialMatrix(X, d); % N x P matrix
P = size(phi_d, 2);                % number of monomials

% data logs
cost_poly_log = zeros(nIter, 1);
mode_error = zeros(nIter, 1);

%% initialization
id_all = randperm(N);
id = id_all(1:2000);
xop = X(id,:);
dz = Xdot(id,:);
true_mode = data.mode(id,:);

% initial polynomial coefficients
a1 = randn(P,1); b1 = randn(P,1);
a2 = randn(P,1); b2 = randn(P,1);

%% Main bilevel cvx optimization loop
for iter = 1:nIter
    phi_sub = phi_d(id,:); % 2000 x P
    H1 = [phi_sub * a1, phi_sub * b1];
    H2 = [phi_sub * a2, phi_sub * b2];
    dy_tilde = {H1, H2};

    %% Step 2: identify soft mode assignments
    prog_mode = spotsosprog;
    %[sol_mode, prog_mode, ~, vars_mode] = searchMode(dz, dy_tilde, prog_mode);  % Linear simplex relaxation
    [sol_mode, prog_mode, ~, vars_mode] = searchMode_moment(dz, dy_tilde, prog_mode); % semidefinite moment relaxation
    lambda_soft = double(sol_mode.eval(vars_mode.lam));

    % tightness check
    tight_count = 0;
    for i = 1:size(lambda_soft,1)
        Mi_num = double(sol_mode.eval(vars_mode.Mi{i}));
        s = svd( (Mi_num + Mi_num.')/2 ); % ensure symmetry
        % Simple spectral gap test for rank-1:
        if numel(s) > 1 && s(2) / max(s(1), 1e-12) < 1e-6
            tight_count = tight_count + 1;
        end
    end
    fprintf('Tight (rank-1) moment matrices: %d / %d\n', tight_count, size(lambda_soft,1));

    [~, mode_op] = max(lambda_soft, [], 2);

    % Label permutation resolution
    if ~isempty(true_mode)
        mismatch1 = nnz(mode_op ~= true_mode);
        mode_op_swapped = 3 - mode_op;
        mismatch2 = nnz(mode_op_swapped ~= true_mode);
        if mismatch2 < mismatch1
            mode_op = mode_op_swapped;
            lambda_soft = lambda_soft(:, [2 1]);
        end
        mode_error(iter) = nnz(mode_op ~= true_mode);
    end

    %% Step 3: fit polynomial dynamics using soft λ for first few iterations
    use_soft_lambda = (iter <= 3);  % Use soft lambda for warm start

    if use_soft_lambda
        lambda_used = lambda_soft;
    else
        lambda_used = full(sparse(1:length(mode_op), mode_op, 1, length(mode_op), Q)); % hard
    end

    prog_poly = spotsosprog;
    [sol_poly, prog_poly, cost_vec, vars_poly] = searchPoly(xop, dz, lambda_used, prog_poly, phi_sub);

    a1 = double(sol_poly.eval(vars_poly.a1));
    b1 = double(sol_poly.eval(vars_poly.b1));
    a2 = double(sol_poly.eval(vars_poly.a2));
    b2 = double(sol_poly.eval(vars_poly.b2));

    cost_poly_log(iter) = double(sol_poly.eval(sum(vars_poly.delta) * 1e3));

    fprintf("Iter %3d | Loss: %.4e | Mode mismatch: %d\n", iter, cost_poly_log(iter), mode_error(iter));
end

%% Switching Surface Recovery
% use same setting as before
kappa = 4;  % degree of switching surface
eta = 10;
epsilon = 1e-2;
beta = 1e-2;

% Construct signed labels from hard mode prediction
sigma = 2 * (mode_op == 1) - 1;

% Optimize separating polynomial surface
[a_opt, z_val, diagnostics] = searchSwitchingSurface_softmargin(xop, sigma, kappa, eta, epsilon, beta);


%% plot
set(groot, 'defaultFigureColorMode', 'manual');
set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesColor', 'w');
set(groot, 'defaultAxesXColor', 'k');
set(groot, 'defaultAxesYColor', 'k');
set(groot, 'defaultAxesZColor', 'k');
set(groot, 'defaultTextColor', 'k');

% Set LaTeX as default font for all plots
set(groot, 'defaultTextInterpreter','latex')
set(groot, 'defaultAxesTickLabelInterpreter','latex')
set(groot, 'defaultLegendInterpreter','latex')

figure(1); clf;
hold on; box on; grid on;
plot(abs(cost_poly_log), '-o', 'LineWidth', 1.5);
plot(1e-10 * ones(nIter,1), 'k--');
xlabel('Iteration'); ylabel('Polynomial ID Loss');
title('Polynomial Identification Convergence'); set(gca, 'YScale', 'log');

% style grid
ax = gca;
ax.GridColor = [0.6 0.6 0.6];
ax.GridAlpha = 0.6;

saveas(gcf, 'fig1.png');
exportgraphics(gcf, 'fig1.pdf', 'ContentType','vector');

figure(2); clf;
hold on; box on; grid on;
plot(mode_error, '-s', 'LineWidth', 1.5);
plot(zeros(nIter,1), 'k--');
xlabel('Iteration'); ylabel('Mismatch Count');
title('Mode Assignment Error');

saveas(gcf, 'fig2.png');
exportgraphics(gcf, 'fig2.pdf', 'ContentType','vector');

%% Plotting Switching Surface & Mode-Labeled Points

% scatter by mode
idx_mode1 = (sigma == +1);  
idx_mode2 = (sigma == -1);

figure(3); clf; hold on; axis equal; grid on;
h1 = scatter(xop(idx_mode1,1), xop(idx_mode1,2), 20, 'y', 'filled', 'DisplayName', 'Mode 1');
h2 = scatter(xop(idx_mode2,1), xop(idx_mode2,2), 20, [0 0 0.8], 'filled', 'DisplayName', 'Mode 2');
legend([h1 h2], 'Location', 'best');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\dot{x}$', 'Interpreter', 'latex');
title('Mode Identification', 'Interpreter', 'latex');

% Evaluate surface
[x1, x2] = meshgrid(linspace(-6,6,200));
Xgrid = [x1(:), x2(:)];
Phi_grid = buildMonomialMatrix(Xgrid, kappa);
Z = Phi_grid * a_opt;
Z = reshape(Z, size(x1));

% plot contour
figure(4); clf; hold on; axis equal; grid on;
h1 = scatter(xop(idx_mode1,1), xop(idx_mode1,2), 20, 'y', 'filled', 'DisplayName', 'Mode 1');
h2 = scatter(xop(idx_mode2,1), xop(idx_mode2,2), 20, [0 0 0.8], 'filled', 'DisplayName', 'Mode 2');
[~, h3] = contour(x1, x2, Z, [0,0], 'k', 'LineWidth', 2);
h3.DisplayName = '$f(x, \dot{x})=0$';

legend([h1 h2 h3], 'Location', 'best', 'Interpreter', 'latex');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\dot{x}$', 'Interpreter', 'latex');
title('Switching Surface $f(x,\dot{x})=0$', 'Interpreter', 'latex');

% Plot switching surface and ground truth energy boundary
figure(5); clf; hold on; axis equal; grid on;

% Scatter colored by mode
h1 = scatter(xop(idx_mode1,1), xop(idx_mode1,2), 20, 'y', 'filled', 'DisplayName', 'Mode 1');
h2 = scatter(xop(idx_mode2,1), xop(idx_mode2,2), 20, [0 0 0.8], 'filled', 'DisplayName', 'Mode 2');

% Grid for contour evaluation
[x1, x2] = meshgrid(linspace(-6,6,300));
Xgrid = [x1(:), x2(:)];

% Identified surface
Phi_grid = buildMonomialMatrix(Xgrid, kappa);  % degree 4
Z_identified = reshape(Phi_grid * a_opt, size(x1));
[~, h3] = contour(x1, x2, Z_identified, [0,0], 'k-', 'LineWidth', 2);
h3.DisplayName = '$f_{\mathrm{identified}}(x,\dot{x}) = 0$';

% True surface
omega = 1.0; lambda = 0.3; Ec = 3.0;
Z_true = 0.5 * x2.^2 + 0.5 * omega^2 * x1.^2 + 0.25 * lambda * x1.^4 - Ec;
[~, h4] = contour(x1, x2, Z_true, [0,0], 'r--', 'LineWidth', 2);
h4.DisplayName = '$f_{\mathrm{true}}(x,\dot{x}) = 0$';

legend([h1 h2 h3 h4], 'Location', 'best', 'Interpreter', 'latex');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\dot{x}$', 'Interpreter', 'latex');
title('Identified vs Ground Truth Switching Surface', 'Interpreter', 'latex');

figure(5);
exportgraphics(gcf,'SPS_switch_surface.pdf','ContentType','vector');
