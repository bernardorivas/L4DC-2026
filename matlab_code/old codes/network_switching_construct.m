clear; clc; close all;

% system parameters
alpha_f = [0.0; -0.1; 0.01];
delta = 9; 
n_samples = 200;
t_log = cell(1, n_samples);
x_log = cell(1, n_samples);
dx_log = cell(1, n_samples);
mode_log = cell(1, n_samples);

% define adjacency matrices and Laplacians
alpha = 1.0; beta = -1.0; % coupling parameters
C = -1; % coupling channel matrix
% graph 1: K3
%A1 = [0 1 1; 1 0 1; 1 1 0];
A1 = [0 1 1; 1 0 1; 1 1 0];
D1 = diag(sum(A1, 2));
L1 = alpha*D1 + beta*A1;

% graph 2: C3
%A2 = [0 1 1; 1 0 0; 1 0 0];
A2 = [0 1 0; 0 0 1; 1 0 0];
D2 = diag(sum(A2, 2));
L2 = alpha*D2 + beta*A2;

% isolated node dynamics f(x) = α₀ + α₁ x + α₂ x²
%f_scalar = @(x) alpha_f(1) + alpha_f(2) * x + alpha_f(3) * x.^2;
%F = @(xvec) arrayfun(f_scalar, xvec); % xvec: joint state vector
F = @(xvec) alpha_f(1) + alpha_f(2)*xvec + alpha_f(3)*(xvec.^2);

% switching rule
%switch_rule = @(xvec) (xvec(2) - xvec(3))^2 > delta;
center = [0; 0; 0];
switch_rule = @(xvec) sum((xvec - center).^2) > delta;

% event function
event_switch = @(t, x) deal(x(1)^2 + x(2)^2 + x(3)^2 - delta, 1, 0);

%% generate trajectories
for k = 1:n_samples
    x0 = (rand(3,1) - 0.5) * 8;
    t0 = 0;
    tf = 3.0;
    t_curr = t0;
    mode = 1 + switch_rule(x0); % initial mode

    t_traj = [];
    x_traj = [];
    dx_traj = [];
    mode_traj = [];

    while t_curr < tf
        if mode == 1
            L = L1;
        else
            L = L2;
        end

        % define full dynamics
        f_full = @(t, x) F(x) + C * L * x;

        if abs(switch_rule(x0) - (mode - 1)) > 1e-3
            fprintf('Warning: mode mismatch at sample %d\n', k);
        end

        options = odeset('Events', @(t, x) switching_event_fn_net(t, x, delta), ...
                         'RelTol', 1e-8, 'AbsTol', 1e-10);
        [t_sol, x_sol, te, xe] = ode89(f_full, t_curr:0.001:tf, x0, options);

        if length(t_sol) < 3
            break;
        end

        dx_sol = zeros(size(x_sol));
        for i = 1:length(t_sol)
            dx_sol(i,:) = f_full(t_sol(i), x_sol(i,:)')';
        end

        % Store data
        t_traj = [t_traj; t_sol];
        x_traj = [x_traj; x_sol];
        dx_traj = [dx_traj; dx_sol];
        mode_traj = [mode_traj; mode * ones(length(t_sol),1)];

        % Check for switch
        if isempty(te)
            break;
        end

        % Prepare next step
        t_curr = te(end);
        x0 = xe(end,:)';
        dt_bump = 1e-4;
        x0 = x0 + dt_bump * dx_sol(end,:)';

        sep = x0(1)^2 + x0(2)^2 - x0(3)^2;
        hyst = 1e-6;
        if mode == 1 && sep > delta + hyst
            mode = 2;
        elseif mode == 2 && sep < delta - hyst
            mode = 1;
        else
            fprintf('Sample %d: Ambiguous switch — terminating early.\n', k);
            break;
        end
    end

    t_log{k} = t_traj;
    x_log{k} = x_traj;
    dx_log{k} = dx_traj;
    mode_log{k} = mode_traj;
end

% Clean and save data
nonempty = ~cellfun(@isempty, x_log);
x_log = x_log(nonempty);
dx_log = dx_log(nonempty);
mode_log = mode_log(nonempty);
t_log = t_log(nonempty);

data.x = cell2mat(x_log');
data.dx = cell2mat(dx_log');
data.mode = cell2mat(mode_log');
save("switching_network_data.mat", "data");


%% Plot Phase Portraits with Switching Surface
figure; hold on;

% --- Trajectories ---
for k = 1:length(x_log)
    plot3(x_log{k}(:,1), x_log{k}(:,2), x_log{k}(:,3), 'b-'); 
    %plot3(x_log{k}(end,1), x_log{k}(end,2), x_log{k}(end,3), '.', 'Color', 'r', 'MarkerSize', 12);
    plot3(x_log{k}(1,1), x_log{k}(1,2), x_log{k}(1,3), '.', 'Color', 'r', 'MarkerSize', 10);
end

% --- Switching Surface (x2 - x3)^2 = delta ---
x1_vals = linspace(-5, 5, 50);
x2_vals = linspace(-5, 5, 50);
x3_vals = linspace(-5, 5, 50);
[X1, X2, X3] = meshgrid(x1_vals, x2_vals, x3_vals);
%F = (X2 - X3).^2 - delta;
F = (X1 - center(1)).^2 + (X2 - center(2)).^2 + (X3 - center(3)).^2 - delta;

h_surf = patch(isosurface(X1, X2, X3, F, 0));
isonormals(X1, X2, X3, F, h_surf);
set(h_surf, 'FaceColor', [0.7 0.5 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.2);

% --- Labels and formatting ---
xlabel('$x_1$', 'Interpreter','latex'); 
ylabel('$x_2$', 'Interpreter','latex'); 
zlabel('$x_3$', 'Interpreter','latex');
title('Phase Portrait of Switching Network System', 'Interpreter','latex');
legend({'Trajectories', 'Initial Points'}, 'Interpreter','latex');

daspect([1 1 1]);
view(135, 30); 
axis tight; 
grid on; 
axis equal;
camlight; lighting gouraud;

% Plot Phase Portraits with Switching Surface and Mode Coloring
figure; hold on;

% --- Switching Surface: Sphere of radius sqrt(delta) ---
x1_vals = linspace(-5, 5, 50);
x2_vals = linspace(-5, 5, 50);
x3_vals = linspace(-5, 5, 50);
[X1, X2, X3] = meshgrid(x1_vals, x2_vals, x3_vals);
F = (X1 - center(1)).^2 + (X2 - center(2)).^2 + (X3 - center(3)).^2 - delta;

h_surf = patch(isosurface(X1, X2, X3, F, 0));
isonormals(X1, X2, X3, F, h_surf);
set(h_surf, 'FaceColor', [0.7 0.5 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.2);

% --- Trajectories colored by mode ---
Q = max(cellfun(@(m) max(m), mode_log));  % number of modes
mode_colors = lines(Q);  % distinct color for each mode

for k = 1:length(x_log)
    x_k = x_log{k};
    mode_k = mode_log{k};

    % Find switching points
    switches = [1; find(diff(mode_k) ~= 0) + 1; length(mode_k)+1];

    for j = 1:length(switches)-1
        idx = switches(j):switches(j+1)-1;
        m_id = mode_k(idx(1));

        plot3(x_k(idx,1), x_k(idx,2), x_k(idx,3), ...
              'Color', mode_colors(m_id,:), 'LineWidth', 1.1);
    end

    % Plot initial point
    %plot3(x_k(1,1), x_k(1,2), x_k(1,3), '.', 'Color', [0.5, 0.9, 0.8], 'MarkerSize', 11);
end

xlabel('$x_1$', 'Interpreter','latex'); 
ylabel('$x_2$', 'Interpreter','latex'); 
zlabel('$x_3$', 'Interpreter','latex');
title('Phase Portrait of Switching Network System', 'Interpreter','latex');

% Construct legend
h_legend = gobjects(Q,1);
for q = 1:Q
    h_legend(q) = plot3(NaN,NaN,NaN, '-', 'Color', mode_colors(q,:), 'LineWidth', 1.5);
end
legend(h_legend, arrayfun(@(q) sprintf('Mode %d', q), 1:Q, 'UniformOutput', false), ...
       'Location', 'northeast', 'Interpreter','latex');

daspect([1 1 1]);
view(135, 30); 
axis tight; 
grid on; 
axis equal;
camlight; lighting gouraud;

figure; hold on;
for k = 1:length(x_log)
    stairs(t_log{k}, mode_log{k}, 'LineWidth', 1.2);
end
xlabel('Time'); ylabel('Mode'); title('Mode Evolution Over Time');

% Define colors and line styles for each mode
mode_colors = lines(7); % for up to 7 modes; use distinguishable colors if needed
mode_styles = {'-', '--', ':', '-.', '-', '--', ':'};

figure; hold on;
N_modes = max(cellfun(@(m) max(m), mode_log));  % number of modes
component = 1;  % which component of x to plot (e.g., x_1)

for k = 1:length(x_log)
    t_k = t_log{k};
    x_k = x_log{k}(:, component);
    mode_k = mode_log{k};
    
    % Find switches
    switches = [1; find(diff(mode_k) ~= 0) + 1; length(mode_k)+1];
    
    for j = 1:length(switches)-1
        idx_range = switches(j):switches(j+1)-1;
        m_id = mode_k(idx_range(1));
        
        plot(t_k(idx_range), x_k(idx_range), ...
            'Color', mode_colors(m_id,:), ...
            'LineStyle', mode_styles{m_id}, ...
            'LineWidth', 1.5);
    end
end

xlabel('Time $t$', 'Interpreter','latex');
ylabel('$x$', 'Interpreter','latex');
title('Component Evolution Colored by Mode', 'Interpreter','latex');

% Legend construction
for m = 1:N_modes
    h_dummy(m) = plot(NaN, NaN, ...
        'Color', mode_colors(m,:), ...
        'LineStyle', mode_styles{m}, ...
        'LineWidth', 1.5);
end
legend(h_dummy, arrayfun(@(m) sprintf('Graph %d', m), 1:N_modes, 'UniformOutput', false), ...
       'Location', 'best', 'Interpreter','latex');
grid on;
ylim padded;


%% Event function
function [value, isterminal, direction] = switching_event_fn_net(t, x, delta)
    %value = (x(2) - x(3))^2 - delta;
    center = [0; 0; 0];
    value = sum((x - center).^2) - delta;
    isterminal = 1;
    direction = 0;
end