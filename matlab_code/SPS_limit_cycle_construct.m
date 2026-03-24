%% sps_switching_limit_cycle_data.m
% Switching Polynomial System (SPS) data generator
%   Mode 1 (|x1| <= 1 - hyst):  xdot = [x2;  mu*(1-x1^2)*x2 - x1^3]
%   Mode 2 (otherwise)        :  xdot = [x2;  mu*(1-x1^2)*x2 - x1  ]
%
% Mirrors the structure of your SLS example (event-based integration,
% logs, plots, and flat-packed 'data' struct).

clear; clc; close all;

%% parameters
mu  = 0.5;     % viscous/van der Pol-like coefficient
t0  = 0; tf = 10;
n_samples = 200;

% region test and (optional) tiny hysteresis
hyst = 0;                           % can be 0; small >0 helps avoid chatter
in_strip = @(x) (abs(x(1)) <= 1 - hyst);

% event: hit either x1=+1 or x1=-1
function [value,isterminal,direction] = switch_events(~,x)
    value      = [x(1)-1; x(1)+1];   % zeros at +1 and -1
    isterminal = [1; 1];             % stop integration
    direction  = [0; 0];             % detect all crossings
end

options = odeset('Events', @switch_events, 'RelTol',1e-8, 'AbsTol',1e-10);

%% logs
t_log    = cell(1,n_samples);
y_log    = cell(1,n_samples);
dy_log   = cell(1,n_samples);
mode_log = cell(1,n_samples);

%% simulate many trajectories (random ICs in [-4,4]^2)
rng(0);
for k = 1:n_samples
    x0 = (rand(2,1)-0.5)*8;   % IC in [-4,4]^2
    tcur = t0;

    while tcur < tf
        if in_strip(x0), mode = 1; else, mode = 2; end
        % integrate current (fixed) mode until switch
        fcur = @(t,x) sps_mode_rhs(x, mode, mu);
        [t_part, y_part, te, xe] = ode45(fcur, [tcur tf], x0, options);

        % evaluate derivatives along the segment
        dy_part = zeros(size(y_part));
        for i = 1:size(y_part,1)
            dy_part(i,:) = sps_mode_rhs(y_part(i,:).', mode, mu).';
        end

        % mode labels along the segment
        mode_part = mode*ones(size(t_part));

        % append logs
        t_log{k}    = [t_log{k};    t_part];
        y_log{k}    = [y_log{k};    y_part];
        dy_log{k}   = [dy_log{k};   dy_part];
        mode_log{k} = [mode_log{k}; mode_part];

        if isempty(te), break; end
        % restart exactly at the hit point
        tcur = te(end);
        x0   = xe(end,:).';
        % tiny push to the new side to avoid re-detecting the same event
        x0(1) = x0(1) + sign(x0(2))*1e-8;
    end
end

%% pack flat dataset (matches your SLS 'data' shape)
data.y    = cell2mat(y_log');      % [sum_i Ti  x  2]
data.dy   = cell2mat(dy_log');     % same size
data.mode = cell2mat(mode_log');   % integer mode labels (1 or 2)
save("sps_limit_cycle_data.mat", "data", "mu", "hyst");

%% plots
figure(1); clf; hold on;
for k=1:n_samples
    plot(y_log{k}(:,1), y_log{k}(:,2), '-');
end
xline( 1,'k--'); xline(-1,'k--'); axis equal
xlabel('$x_1$', 'Interpreter','latex'); ylabel('$x_2$', 'Interpreter','latex');
title('SPS: phase portrait with switching at $|x_1|=1$','Interpreter','latex'); grid on;

figure(2); clf; hold on;
for k=1:n_samples, stairs(t_log{k}, mode_log{k}, '-'); end
xlabel('Time', 'Interpreter','latex'); ylabel('Mode', 'Interpreter','latex'); title('Mode Switching', 'Interpreter','latex'); grid on;

% Energy-like diagnostic (omega=1)
omega = 1;
figure(3); clf;
for k=1:n_samples
    E = 0.5*y_log{k}(:,2).^2 + 0.5*omega^2*y_log{k}(:,1).^2;
    subplot(3,1,1); hold on; plot(y_log{k}(:,1), y_log{k}(:,2), '-');
    subplot(3,1,2); hold on; plot(t_log{k}, y_log{k});
    subplot(3,1,3); hold on; plot(t_log{k}, E);
end
subplot(3,1,1); xlabel('x_1'); ylabel('x_2'); title('Trajectories');
subplot(3,1,2); xlabel('t'); ylabel('x');
subplot(3,1,3); xlabel('t'); ylabel('Energy'); set(gca,'YScale','log'); grid on;

%% Ground-truth mode scatter checks (color by true mode)
% Flatten once (already done above as 'data')
X_all   = data.y;          % [N x 2]
m_all   = data.mode(:);    % [N x 1], 1 or 2

idx1 = (m_all == 1);
idx2 = (m_all == 2);

% --- (4) Scatter in state space colored by true mode ---
figure(4); clf; hold on; axis equal; grid on;
h1 = scatter(X_all(idx1,1), X_all(idx1,2), 12, 'y', 'filled', 'DisplayName','Mode 1 (|x_1|<1)');
h2 = scatter(X_all(idx2,1), X_all(idx2,2), 12, [0 0 0.8], 'filled', 'DisplayName','Mode 2 (|x_1|\ge 1)');
xline( 1,'k--','LineWidth',1.5,'DisplayName','|x_1|=1');
xline(-1,'k--','LineWidth',1.5,'HandleVisibility','off');
xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex');
title('Ground-truth mode scatter in $(x_1,x_2)$','Interpreter','latex');
legend('Location','best','Interpreter','latex');

% --- (5) Same scatter but lightly transparent to reveal density (if supported) ---
figure(5); clf; hold on; axis equal; grid on;
s1 = scatter(X_all(idx1,1), X_all(idx1,2), 12, 'y', 'filled', 'DisplayName','Mode 1'); 
s2 = scatter(X_all(idx2,1), X_all(idx2,2), 12, [0 0 0.8], 'filled', 'DisplayName','Mode 2');
try
    s1.MarkerFaceAlpha = 0.4; s1.MarkerEdgeAlpha = 0.2;
    s2.MarkerFaceAlpha = 0.4; s2.MarkerEdgeAlpha = 0.2;
end
xline( 1,'k--','LineWidth',1.5);
xline(-1,'k--','LineWidth',1.5);
xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex');
title('Mode scatter with transparency (density cue)','Interpreter','latex');
legend('Location','best','Interpreter','latex');

% --- (6) Quick sanity: counts per mode ---
figure(6); clf; box on; grid on;
counts = [nnz(idx1), nnz(idx2)];
bar([1 2], counts, 0.6);
set(gca,'XTick',[1 2],'XTickLabel',{'Mode 1','Mode 2'});
ylabel('Number of samples'); title('Mode sample counts (ground truth)');
text(1, counts(1), sprintf('%d',counts(1)), 'HorizontalAlignment','center','VerticalAlignment','bottom');
text(2, counts(2), sprintf('%d',counts(2)), 'HorizontalAlignment','center','VerticalAlignment','bottom');

% --- (7) Optional: trajectory-wise scatter, colored by true mode (subset for clarity) ---
figure(7); clf; hold on; axis equal; grid on;
showK = min(20, n_samples);           % plot first 20 trajectories for clarity
for k = 1:showK
    Yk = y_log{k}; Mk = mode_log{k};
    plot(Yk(Mk==1,1), Yk(Mk==1,2), '.', 'Color',[0.929,0.694,0.125]); % yellow-ish
    plot(Yk(Mk==2,1), Yk(Mk==2,2), '.', 'Color',[0    ,0    ,0.8  ]); % blue
end
xline( 1,'k--','LineWidth',1.5);
xline(-1,'k--','LineWidth',1.5);
xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex');
title('Per-trajectory ground-truth mode coloring (subset)','Interpreter','latex');

%% ---- local function(s) ----
function dx = sps_mode_rhs(x, mode, mu)
% Piecewise polynomial field:
% mode 1 (inside strip):   x' = [x2; mu*(1-x1^2)*x2 - x1^3]
% mode 2 (outside strip):  x' = [x2; mu*(1-x1^2)*x2 - x1  ]
    x1 = x(1); x2 = x(2);
    if mode == 1
        xterm = x1^3;
    else
        xterm = x1;
    end
    dx = [x2; mu*(1 - x1^2)*x2 - xterm];
end
