clear; clc; close all;

% system parameters for each mode
params1 = struct('gamma', 0.1, 'a', 0.2, 'b', 0.3, 'c', -0.1);
params2 = struct('gamma', 1.4, 'a', 0.1, 'b', 0.1, 'c', -0.05);
lambda = 0.3;

Ec = 3.0; % energy threshold
n_samples = 300;
t_log = cell(1, n_samples);
y_log = cell(1, n_samples);
dy_log = cell(1, n_samples);
mode_log = cell(1, n_samples);

% quartic oscillator dynamics, y: state, p: parameters
%dyn = @(y, p) [y(2); -p.gamma * y(2) - 0.1 * y(2)^3 - lambda * y(1)^3 + p.a * y(1)^2 + p.b * y(1) * y(2) + p.c];

energy = @(y) 0.5 * y(2)^2 + 0.5 * y(1)^2 + (lambda/4) * y(1)^4;
switch_rule = @(y) energy(y) > Ec;

% Define switching event function (like in the linear system)
event_switch = @(t, y) deal(energy(y) - Ec, 1, 0); % 1: isterminal, 0: direction


for k = 1:n_samples
    y0 = (rand(2,1) - 0.5) * 9;
    t0 = 0;
    tf = 20;
    t_curr = t0;
    mode = 1 + switch_rule(y0);

    t_traj = [];
    y_traj = [];
    dy_traj = [];
    mode_traj = [];

    while t_curr < tf
        %if switch_rule(y0) == 1
        if mode == 2
            f = @(t, y) [y(2); - y(1) - 0.1 * y(2)^3 - lambda * y(1)^3 + 0.2 * y(1)^2 + 0.3 * y(1) * y(2) - 0.1];
        elseif mode == 1
            f = @(t, y) [y(2); - y(1) - lambda * y(1)^3 - 0.1 * y(2)^3 + 0.1 * y(1) * y(2) + 0.1*y(1)^2 - 0.1*y(2)];
        end

        if abs(switch_rule(y0) - (mode - 1)) > 1e-3
            % this logs if energy and mode disagree *after* simulation
            fprintf('Warning: mode-energy mismatch after sample %d\n', k);
        end

        
        % integrate dynamics
        switch_event = @(t, y) switching_event_fn(t, y, energy, Ec);
        options = odeset('Events', switch_event, 'RelTol', 1e-8, 'AbsTol', 1e-10);
        [t_sol, y_sol, te, ye] = ode89(f, [t_curr tf], y0, options);
        
        % skip short or empty segments
        if length(t_sol) < 3
            break;  % skip segment too small (possibly bad switch)
        end
        
        % compute derivatives
        dy_sol = zeros(size(y_sol));
        for i = 1:length(t_sol)
            dy_sol(i,:) = f(t_sol(i), y_sol(i,:)')';
        end
        
        % store data
        t_traj = [t_traj; t_sol];
        y_traj = [y_traj; y_sol];
        dy_traj = [dy_traj; dy_sol];
        mode_traj = [mode_traj; mode * ones(size(t_sol))];

        % stop if no switch
        if isempty(te)
            break;
        end
        

        if size(ye, 1) > 1
            warning("Multiple switching events detected at once; using last one.");
        end
        t0 = te(end);  % start over from t = event time
        y0 = ye(end, :);

        dt_bump = 1e-5;
        y0 = y0 + dt_bump * dy_sol(end, :);  % Update state for next iteration
        E_post = energy(y0);
        hyst = 0.00001;

        if mode == 1 && E_post > Ec + hyst
            mode = 2;
        elseif mode == 2 && E_post < Ec - hyst
            mode = 1;
        else
            % Event triggered but not a valid switch; likely glancing or numerical noise
            fprintf("Sample %d: Ambiguous switch — terminating early.\n", k);
            break;
        end
        %mode = switch_rule(y0) + 1;
        %mode = 3 - mode;
    end

    t_log{k} = t_traj;
    y_log{k} = y_traj;
    dy_log{k} = dy_traj;
    mode_log{k} = mode_traj;
end

% Remove empty entries before saving/plotting
nonempty_idx = ~cellfun(@isempty, y_log);
y_log = y_log(nonempty_idx);
dy_log = dy_log(nonempty_idx);
mode_log = mode_log(nonempty_idx);
t_log = t_log(nonempty_idx);

data.y = cell2mat(y_log');
data.dy = cell2mat(dy_log');
data.mode = cell2mat(mode_log');
save("switching_quartic_oscillator_data.mat", "data");

%% Plotting phase space
figure;
hold on;
for k = 1:n_samples
    plot(y_log{k}(:,1), y_log{k}(:,2), 'b-');  % use lines instead of dots
end
xlabel('$\phi$', 'Interpreter', 'latex');
ylabel('$\dot{\phi}$', 'Interpreter', 'latex');
title('Phase Portrait of Switching Quartic Oscillator', 'Interpreter', 'latex');

% Define grid and energy contour
[PHI, PHIDOT] = meshgrid(linspace(-4, 4, 400), linspace(-4, 4, 400));
ENERGY = 0.5 * PHIDOT.^2 + 0.5 * PHI.^2 + (lambda / 4) * PHI.^4;

% Overlay switching surface
contour(PHI, PHIDOT, ENERGY, [Ec Ec], 'r--', 'LineWidth', 2, 'DisplayName', '$E = E_c$');
grid on;
axis equal;

figure;
hold on;

% --- Plot trajectories by mode ---
for k = 1:n_samples
    yk = y_log{k};
    mk = mode_log{k};

    switches = find(diff(mk) ~= 0);
    segment_start = 1;
    for j = [switches(:)' length(mk)]  % include final segment
        idx = segment_start:j;
        if mk(segment_start) == 1
            plot(yk(idx,1), yk(idx,2), 'b');  % Mode 1
        else
            plot(yk(idx,1), yk(idx,2), 'r');  % Mode 2
        end
        segment_start = j + 1;
    end
end

xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\dot{x}$', 'Interpreter', 'latex');
title('Phase Portrait of Switching Quartic Oscillators', 'Interpreter', 'latex');

% --- Add switching surface (E = Ec) ---
[PHI, PHIDOT] = meshgrid(linspace(-6.0, 6.0, 400), linspace(-6.0, 6.0, 400));
ENERGY = 0.5 * PHIDOT.^2 + 0.5 * PHI.^2 + (lambda / 4) * PHI.^4;
[~, h_switch] = contour(PHI, PHIDOT, ENERGY, [Ec Ec], 'k--', 'LineWidth', 2);

% --- Dummy handles for legend entries ---
h_mode1 = plot(NaN, NaN, 'b');
h_mode2 = plot(NaN, NaN, 'r');

legend([h_mode1, h_mode2, h_switch], ...
       {'Mode 1', 'Mode 2', 'Switching Surface'}, ...
       'Interpreter', 'latex', 'Location', 'best');

axis equal;
grid on;

figure;
for k = 1:n_samples
    stairs(t_log{k}, mode_log{k});
    hold on;
end
title('Mode Evolution Over Time'); xlabel('Time'); ylabel('Mode');


function [value, isterminal, direction] = switching_event_fn(t, y, energy, Ec)
    value = energy(y) - Ec;
    isterminal = 1;       % stop the integration at the event
    direction = 0;        % detect all zero crossings
end