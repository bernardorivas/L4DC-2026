clear all;clc;close all;
%%
syms x [2, 1] 'real'
omega = 1.0;
gamma1 = 0.1; % damping coefficient 1
gamma2 = 0.5; % damping coefficient 2

A1 = [0 1; -omega^2 -gamma1];
A2 = [0 1; -omega^2 -gamma2];

% Switching function
switch_rule = @(x) x(1) < 0;  % switch depending on position (instead of velociy)


% Simulate many sample trajectories
n_samples = 200; % number of trajectories
t_log = cell(1, n_samples);
y_log = cell(1, n_samples);
dy_log = cell(1, n_samples);
mode_log = cell(1, n_samples);

%%
for k = 1:n_samples
    % Random initial condition
    x0 = (rand(2, 1) - 0.5) * 10;  % sample from [-5, 5]^2
    tf = 10;
    t0 = 0;

    % Determine initial mode
    mode = switch_rule(x0) + 1; % mode = 1 (logic 0): x >= 0, mode = 2 (logic 1): x < 0

    options = odeset('Events', @linear_ho_event_switch);

    while t0 < tf
        if switch_rule(x0) == 0
            A = A1;
        elseif switch_rule(x0) == 1
            A = A2;
        end

        [t_partial, y_partial, te, xe] = ode89(@(t, x) A*x, t0:0.0003:tf, x0, options);
        dy_partial = (A * y_partial')'; % compute derivatives for all times
        mode_partial = mode * ones(length(t_partial), 1);

        t_log{k} = [t_log{k}; t_partial];
        y_log{k} = [y_log{k}; y_partial];
        dy_log{k} = [dy_log{k}; dy_partial];
        mode_log{k} = [mode_log{k}; mode_partial];

        if isempty(te)
            break
        end

        t0 = te;  % start over from t = event time
        if size(xe, 1) > 1
            warning("Multiple switching events detected at once; using last one.");
        end
        x0 = xe(end, :); % start over from x = event location

        %x0 = xe;  % start over from x = event location
        %mode = 3 - mode; % switch mode
        mode = switch_rule(x0) + 1;
    end

end

data.y = cell2mat(y_log');
data.dy = cell2mat(dy_log');
data.mode = cell2mat(mode_log');

save("switching_ho_data.mat", "data")

%%

figure()
% phase space trajectories
for k = 1:n_samples
    plot(y_log{k}(end, 1), y_log{k}(end, 2), 'o')
    plot(y_log{k}(:, 1), y_log{k}(:, 2), '-')
    hold on;
end
xlabel('$x_1$ (position)', 'Interpreter', 'latex')
ylabel('$x_2$ (velocity)', 'Interpreter', 'latex')
title('Phase Space Trajectory: $(x_1, x_2)$', 'Interpreter', 'latex')

%%
figure(2)
for k = 1:n_samples

    E = 0.5 * y_log{k}(:,2).^2 + 0.5 * omega^2 * y_log{k}(:,1).^2;

    subplot(3, 1, 1)
    hold on
    plot(y_log{k}(end, 1), y_log{k}(end, 2), 'o')
    plot(y_log{k}(:, 1), y_log{k}(:, 2), '-')
    % daspect([1, 1 ,1])

    % ylim([-2, 2])

    subplot(3, 1, 2); hold on;
    plot(t_log{k}, y_log{k}) 

    subplot(3, 1, 3); hold on;
    plot(t_log{k}, E);  ylabel('Energy'); xlabel('Time');
    set(gca, 'YScale', 'log')
end

%%
figure(3); clf;
for k = 1:n_samples
    stairs(t_log{k}, mode_log{k}(:), "-o"); hold on;
end
xlabel('Time'); ylabel('Mode');
title('Mode Switching');