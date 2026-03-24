%% sls_bistability_construct.m
% Switching Linear System (SLS) with Bistability
% ---------------------------------------------------------------
%   x = (x1, x2)
%   Modes (i,j) ∈ {0,1}^2 determined by thresholds T1, T2
%   Dynamics:   xdot = A*x + b_ij
%   where A = -diag(gamma1, gamma2)
%
%   b_ij = [ L1 + j*(U1-L1);
%            L2 + i*(U2-L2) ],
%   with i = H(T1 - x1), j = H(T2 - x2), and H(0)=1.
%
%   This system exhibits bistability similar to a gene toggle.
%
% ---------------------------------------------------------------

clear; clc; close all;

%% ===== PARAMETERS (fill in concrete numbers later) =====
gamma1 = 1.0;      % e.g. 0.8
gamma2 = 1.0;      % e.g. 0.8
T1     = 3.0;      % e.g. 1.0
T2     = 3.0;      % e.g. 1.0
L1     = 1.0;      % e.g. 0.2
L2     = 1.0;      % e.g. 0.2
U1     = 5.0;      % e.g. 1.2
U2     = 5.0;      % e.g. 1.2
hyst   = 0.01;      % e.g. 0.01
t0     = 0;
tf     = 1;      % e.g. 20
n_samples = 200;   % e.g. 200

%% ===== DEFINE MATRICES AND HELPERS =====
A = -diag([gamma1, gamma2]);
H = @(s) double(s >= 0);

% mode indexing
ij_to_k = @(i,j) 1 + i + 2*j;  % k ∈ {1,2,3,4}

% determine mode for given x
mode_fun = @(x) ij_to_k( H(T1 - x(1)), H(T2 - x(2)) );

% production vector b(x)
b_fun = @(x) [L1 + H(T2 - x(2))*(U1 - L1);
              L2 + H(T1 - x(1))*(U2 - L2)];

% vector field
f = @(x) A*x + b_fun(x);

%% ===== EVENT FUNCTION FOR SWITCHING =====
function [value,isterminal,direction] = switch_events(~,x,T1,T2,h)
    value = [x(1)-(T1-h); x(1)-(T1+h); x(2)-(T2-h); x(2)-(T2+h)];
    isterminal = [1;1;1;1];
    direction  = [0;0;0;0];
end

options = odeset('Events', @(t,x)switch_events(t,x,T1,T2,hyst), ...
                 'RelTol',1e-8,'AbsTol',1e-10);

%% ===== LOGS =====
t_log    = cell(1,n_samples);
y_log    = cell(1,n_samples);
dy_log   = cell(1,n_samples);
mode_log = cell(1,n_samples);

%% ===== SIMULATION LOOP =====
rng(0);
for k = 1:n_samples
    x0 = [3;3] + 4*(rand(2,1) - 0.5);  % initial condition in [1,5]^2
    tcur = t0;

    while tcur < tf
        mode_k = mode_fun(x0);
        fcur = @(t,x) f(x);

        [t_part, y_part, te, xe] = ode45(fcur, [tcur tf], x0, options);

        % compute derivatives and modes along trajectory
        dy_part = zeros(size(y_part));
        mode_part = zeros(size(y_part,1),1);
        for i = 1:size(y_part,1)
            xi = y_part(i,:)';
            dy_part(i,:) = f(xi)';
            mode_part(i) = mode_fun(xi);
        end

        % append logs
        t_log{k}    = [t_log{k};    t_part];
        y_log{k}    = [y_log{k};    y_part];
        dy_log{k}   = [dy_log{k};   dy_part];
        mode_log{k} = [mode_log{k}; mode_part];

        if isempty(te), break; end
        % restart slightly past the event
        tcur = te(end);
        x0   = xe(end,:)';
        x0 = x0 + 1e-8*sign(randn(2,1));
    end
end

%% ===== PACK DATASET =====
data.y    = cell2mat(y_log');
data.dy   = cell2mat(dy_log');
data.mode = cell2mat(mode_log');
save("sls_bistability_data.mat","data");

%% ===== PLOTS =====
% Phase portrait
figure(1); clf; hold on; grid on; axis equal;
for k=1:n_samples
    plot(y_log{k}(:,1), y_log{k}(:,2), '-');
end
xline(T1,'k--'); yline(T2,'k--');
xlabel('$x_1$','Interpreter','latex');
ylabel('$x_2$','Interpreter','latex');
title('SLS bistability: trajectories and thresholds','Interpreter','latex');

% Mode scatter
X_all = data.y; m_all = data.mode;
colors = [0.9 0.7 0.1; 0 0.5 0.8; 0.5 0 0.5; 0 0.8 0.2];
figure(2); clf; hold on; grid on; axis equal;
for k = 1:4
    idx = (m_all == k);
    scatter(X_all(idx,1), X_all(idx,2), 12, colors(k,:), 'filled');
end
xline(T1,'k--'); yline(T2,'k--');
xlabel('$x_1$','Interpreter','latex');
ylabel('$x_2$','Interpreter','latex');
title('Mode regions (true labels)','Interpreter','latex');
legend({'Mode 1','Mode 2','Mode 3','Mode 4'},'Location','best','Interpreter','latex');

% Mode counts
figure(3); clf; box on; grid on;
counts = arrayfun(@(kk) nnz(m_all==kk), 1:4);
bar(1:4, counts, 0.6);
set(gca,'XTick',1:4,'XTickLabel',{'1','2','3','4'});
ylabel('Number of samples');
xlabel('Mode index');
title('Mode sample counts');

%% ===== LOCAL FUNCTION DEFINITIONS =====
function dx = bistable_rhs(x, A, L1,L2,U1,U2,T1,T2)
    H = @(s) double(s>=0);
    b = [L1 + H(T2-x(2))*(U1-L1);
         L2 + H(T1-x(1))*(U2-L2)];
    dx = A*x + b;
end
