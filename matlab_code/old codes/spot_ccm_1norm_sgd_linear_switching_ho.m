%% bilevel switchig linear system identification
% dx/dt = lambda1*A1*x + lambda2*A2*x with lambda1 + lambda2 = 1
% Goal: identify A1, A2 and lambda using bilevel optimization

clear;clc;close all;

load("switching_ho_data.mat") % load data.y, data.dy, data.mode
X = data.y;     % N x 2
Xdot = data.dy; % N x 2
N = size(X, 1);
n = 2; % state dimension
Q = 2; % number of modes

%% logs of the cost
nIter = 10;
cost_matrix_log = zeros(nIter,1);
mode_error = zeros(nIter,1);

%% initialize

% do not add noise for sanity check.
% add small noise to verifyu convergence

id = randperm(N);
id = id(1:2000);

data_raw = data;
%data.lambda = rand(N,1); % random soft initial lambda

%% Main iteration loop
for iter = 1:nIter
    % sample batch
    id = randperm(N,2000);
    xop = X(id,:); % true states (sampled states)
    dz = Xdot(id, :); % true velocities (sampled velocities)
    true_mode = data_raw.mode(id,:); % true mode

    %% LP-relaxation to identify the mode
    
    % initialize matrices 1 and 2
    if iter == 1,  A1 = eye(2); A2 = eye(2); end

    % construct expected vector fields for each mode
    % dy_tilde{1} = A1*x, dy_tilde{2} = A2*x
    dy_tilde = cell(1, Q);
    for q = 1:Q
        A = (q == 1) * A1 + (q == 2) * A2; % if n > 2, add more terms
        dy_q = zeros(size(xop));
        for k = 1:size(xop, 1)
            dy_q(k, :) = double(A * xop(k, :)');
        end
        dy_tilde{q} = dy_q;
    end

    %% Search for lambda (mode assignment)
    prog_mode = spotsosprog;
    %[sol_mode, prog_mode, ~, vars_mode] = searchMode(dz, dy_tilde, prog_mode); % dz: true velocities, 
    [sol_mode, prog_mode, ~, vars_mode] = searchMode_moment(dz, dy_tilde, prog_mode);

    lambda_soft = double(sol_mode.eval(vars_mode.lam));  % N x 2 matrix soft mode
    % tihgtness check
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

    [~, mode_op] = max(lambda_soft, [], 2);  % N x 1 column vector hard mode

    % this part is not necessary but it handles potential label-switching symmetry issue
    if ~isempty(true_mode)
        % Compute mismatch under current mode labels
        mismatch1 = nnz(mode_op ~= true_mode);
        % Compute mismatch under swapped labels
        mode_op_swapped = 3 - mode_op;  % since mode ∈ {1,2}
        mismatch2 = nnz(mode_op_swapped ~= true_mode);

        % Choose the better one
        if mismatch2 < mismatch1
            mode_op = mode_op_swapped;
            lambda_soft = lambda_soft(:, [2 1]);  % swap columns
        end
        % Log mode mismatch if ground truth exist
        mode_error(iter) = nnz(mode_op ~= true_mode);  % now should drop to 0
    end

    % Reconstruct hard lambda from mode_op
    lambda_hard = full(sparse(1:length(mode_op),mode_op,1,length(mode_op),Q));
    %% Search A1, A2 via fixed lambda
    prog_matrix = spotsosprog;
    [sol_matrix, prog_matrix, cost_vec, vars_matrix] = searchMatrix(xop, dz, lambda_hard, prog_matrix);

    cost_matrix_log(iter) = double(sol_matrix.eval(sum(vars_matrix.delta) * 1e3));
    
    % store A1, A2 separately
    A1 = reshape(double(sol_matrix.eval(vars_matrix.a1)), n, n);
    A2 = reshape(double(sol_matrix.eval(vars_matrix.a2)), n, n);

    %% display progress
    fprintf("Iter %3d | Loss: %.4e | Mode mismatch: %d\n", iter, cost_matrix_log(iter), mode_error(iter));
    
end

%% Switching surface recovery

% kappa_surface = 2;
% [prog_base, a, ~, ~] = getPolyConstraints(n, kappa_surface);
% 
% %  construct sigma from identified modes
sigma = 2 * (mode_op == 1) - 1; % convert mode labels to ±1
% 
% % switching surface optimization
% epsilon = 1e-2;
% beta = 1e-2;
% eta = 10;
% [sol, prog, cost_vec] = searchSwitchingSurface(xop, sigma, kappa_surface, prog_base, a, epsilon, beta, eta);
% 
% pnum = sol.eval(a);

% Inputs
kappa = 2;
eta = 10;
epsilon = 1e-2;
beta = 1e-2;
M = 10;

%[a_opt, z_val, diagnostics] = searchSwitchingSurface_MILP(xop, sigma, kappa, eta, epsilon, M);
[a_opt, z_val, diagnostics] = searchSwitchingSurface_softmargin(xop, sigma, kappa, eta, epsilon, beta);


%% Plotting

% Set LaTeX as default font for all plots
set(groot, 'defaultTextInterpreter','latex')
set(groot, 'defaultAxesTickLabelInterpreter','latex')
set(groot, 'defaultLegendInterpreter','latex')

% Matrix identification convergence
h1 = figure(1); clf;
hold on; box on; grid on;

plot(abs(cost_matrix_log), '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
plot(1e-10 * ones(nIter,1), 'k--');
xlabel("Iteration", 'FontSize', 14);
ylabel("Matrix ID Loss", 'FontSize', 14);
title("Matrix Identification Convergence", 'FontSize', 16);
set(gca, 'YScale', 'log');
legend("Loss", "Threshold", 'Location', 'best');

% Mode classification error
h2 = figure(2); clf;
hold on; box on; grid on;

plot(mode_error, '-s', 'LineWidth', 1.5, 'MarkerSize', 6);
plot(zeros(nIter,1), 'k--');
xlabel("Iteration", 'FontSize', 14);
ylabel("Mismatch Count", 'FontSize', 14);
title("Mode Assignment Error", 'FontSize', 16);
legend("Mismatch", "Perfect Classification", 'Location', 'best');



%% sanity check

figure(3)
sigma = 2 * (mode_op == 1) - 1; % convert mode labels to ±1
scatter(xop(:,1), sigma, 'filled'); grid on;
xlabel('$x_1$','Interpreter','latex');
ylabel('Mode $\sigma$','Interpreter','latex');
title('Mode vs $x_1$');

idx_mode1 = (sigma == +1);  
idx_mode2 = (sigma == -1);

figure(4); hold on; axis equal;

h1 = scatter(xop(idx_mode1,1), xop(idx_mode1,2), 20, 'y', 'filled', 'DisplayName', 'Mode 1');
h2 = scatter(xop(idx_mode2,1), xop(idx_mode2,2), 20, [0 0 0.8], 'filled', 'DisplayName', 'Mode 2');

legend([h1 h2], 'Location', 'best');
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
title('Mode Identification', 'Interpreter', 'latex');
grid on;

%% plot identified surface
% Evaluate fitted surface
Phi = buildMonomialMatrix(xop, kappa);
fx = Phi * a_opt;
predicted_signs = sign(fx);

% Evaluate on grid
[x1, x2] = meshgrid(linspace(-6,6,200));
Xgrid = [x1(:), x2(:)];
Phi_grid = buildMonomialMatrix(Xgrid, kappa);
Z = Phi_grid * a_opt;
Z = reshape(Z, size(x1));

% Plot
figure(5); clf; hold on; axis equal; grid on;

h1 = scatter(xop(idx_mode1,1), xop(idx_mode1,2), 10, 'y', 'filled', 'DisplayName', 'Mode 1');
h2 = scatter(xop(idx_mode2,1), xop(idx_mode2,2), 10, [0 0 0.8], 'filled', 'DisplayName', 'Mode 2');
[~, h3] = contour(x1, x2, Z, [0,0], 'k', 'LineWidth', 2);  % capture contour handle
h3.DisplayName = '$f(x)=0$';  % set legend label

% Legend (compact)
leg = legend([h1 h2 h3], 'Location', 'best', 'Interpreter', 'latex');
set(leg, 'FontSize', 8, 'ItemTokenSize', [8, 4]);  % smaller font and symbol size

xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 10);
title('Switching Surface $f_{\mathrm{identified}}(x)=0$', 'Interpreter', 'latex', 'FontSize', 12);

% Figure size and export
set(gcf, 'Units', 'inches', 'Position', [1 1 5 3]);  % width=5in, height=3in
exportgraphics(gcf, 'SLS_switch_surface.pdf', 'ContentType', 'vector');


%%
% figure;
% semilogy(abs(cost_matrix_log), '-o');
% grid on;
% xlabel('Iteration'); ylabel('Matrix ID Loss (1-norm)');
% set(gca, 'YScale', 'log')
% title('Matrix Identification Convergence');
% 
% figure;
% plot(mode_error, '-');
% grid on;
% xlabel('Iteration'); ylabel('Mode Mismatch Count');
% title('Mode Classification Error');

% h = figure(1);
% 
% hold on
% box on
% grid on
% % cost_num = double(sol_lp.eval(lp_vec));
% cost_num = double(sol_sdp.eval(sdp_vec));
% 
% err_list = ["$\delta_x$", "$\delta_y$", "$\delta_z$"];
% 
% for k = 1:3
%     subplot(3, 1, k)
%     % plot(cost_num(k:3:end))
%     plot(-sort(-abs(cost_num(k:3:end))))
%     hold on
%     plot(cost_num(k:3:end) * 0 + 1e-3, 'k--')
%     if k == 1
%         title("Element-wise regression loss")
%     end
% 
%     if k == 3
%         xlabel("Time(s)")
%     end
% 
%     ylabel(err_list(k), "Interpreter", "latex", "FontSize", 14)
%     set(gca, 'YScale', 'log')
%     box on
%     grid on
% end
% 
% set(h, 'Units','pixels');
% % set(h, 'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize', fig_size)
% % print("fig1", "-dpdf")
%%
% h = figure(2);
% 
% subplot(2, 1, 1)
% hold on
% box on
% grid on
% % plot(cost_sdp / 1e4 / 6000, '-o')
% % plot(cost_lp / 1e4 / 6000, '-o')
% plot(abs(cost_sdp))
% % plot(abs(cost_lp))
% set(gca, 'YScale', 'log')
% % set(gca, 'XScale', 'log')
% 
% xlabel("Iteration")
% ylabel("Loss")
% legend({"Sys 1", "Sys 2"})

% subplot(2, 1, 2)
% hold on
% box on
% grid on
% cost_all = [cost_sdp,cost_lp];
% cost_all(1:2:end) = cost_sdp;
% cost_all(2:2:end) = cost_lp;
% plot((cost_all / 1e4 / 6000), '-o')
% set(gca, 'YScale', 'log')
% set(gca, 'XScale', 'log')




% set(h, 'Units','pixels');
% set(h, 'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize', fig_size)
% print("fig2", "-dpdf")


%
% h =figure(3)
% Jw = data.w;% * data.J;
% plot3(Jw(:, 1), Jw(:, 2), Jw(:, 3), '.')
% box on
% grid on
% xlabel("$L_x$", "Interpreter","latex", "FontSize", 14)
% ylabel("$L_y$", "Interpreter","latex", "FontSize", 14)
% zlabel("$L_z$", "Interpreter","latex", "FontSize", 14)
%
% set(h, 'Units','pixels');
% % set(h, 'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize', fig_size)
% % print("fig3", "-dpdf")