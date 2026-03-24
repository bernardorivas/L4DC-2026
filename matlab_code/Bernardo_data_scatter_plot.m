clc; clear all; close all;
% Load the data (assuming in current folder)
load('piecewise_vdp_data.mat');

% --- Basic scatter plot of data samples ---
figure(1); clf; hold on; grid on; axis equal;
scatter(X(:,1), X(:,2), 25, 'filled', 'MarkerFaceAlpha', 0.4);
xlabel('$x_1$', 'Interpreter','latex');
ylabel('$x_2$', 'Interpreter','latex');
title('Sampled states X (500 points)');
set(gca,'FontSize',12);

% --- Quiver plot of vector field (on X_field) ---
figure(2); clf; hold on; grid on; axis equal;
quiver(X_field(:,1), X_field(:,2), F_field(:,1), F_field(:,2), 'AutoScale','on', 'Color',[0 0.4 0.8]);
xlabel('$x_1$', 'Interpreter','latex');
ylabel('$x_2$', 'Interpreter','latex');
title('Vector field samples $(X\_{field}, F\_{field})$', 'Interpreter','latex');
set(gca,'FontSize',12);

% --- Overlay both (optional) ---
figure(3); clf; hold on; grid on; axis equal;
scatter(X(:,1), X(:,2), 20, [0.8 0.6 0], 'filled', 'DisplayName','X samples');
quiver(X_field(:,1), X_field(:,2), F_field(:,1), F_field(:,2), 'Color',[0 0.4 0.8], 'DisplayName','Field');
xlabel('$x_1$', 'Interpreter','latex');
ylabel('$x_2$', 'Interpreter','latex');
title('Training samples and vector field overlay', 'Interpreter','latex');
legend('Location','best');
set(gca,'FontSize',12);

%% ===================== MY DATA (point–velocity pairs) =====================
% File: sps_limit_cycle_data.mat  (adjust name if needed)
% Expected: either data.y/data.dy  OR  X/Xdot  OR  y/dy

try
    S = load('sps_limit_cycle_data.mat');
catch
    error('Could not find sps_limit_cycle_data.mat in current folder.');
end

% --- extract (X_my, V_my) in RAW coordinates ---
if isfield(S,'data') && isstruct(S.data) && isfield(S.data,'y') && isfield(S.data,'dy')
    X_my = S.data.y;     % N x 2 states
    V_my = S.data.dy;    % N x 2 velocities
elseif isfield(S,'X') && isfield(S,'Xdot')
    X_my = S.X;          % N x 2
    V_my = S.Xdot;       % N x 2
elseif isfield(S,'y') && isfield(S,'dy')
    X_my = S.y;          % N x 2
    V_my = S.dy;         % N x 2
else
    error(['sps_limit_cycle_data.mat does not contain recognizable fields.\n' ...
           'Expected (data.y,data.dy) or (X,Xdot) or (y,dy).']);
end

% --- optional: if your file stores normalized velocities, undo it here
% (comment out if not applicable)
% if isfield(S,'md') && isfield(S,'sd')
%     V_my = V_my .* S.sd + S.md;
% end

% --- subsample for clarity ---
rng(42);                                     % reproducible selection
N_all  = size(X_my,1);
N_show = min(1000, N_all);                    % change to 500–2000 as you prefer
idx    = randperm(N_all, N_show);
Xs     = X_my(idx,:);                         % states (x1,x2)
Vs     = V_my(idx,:);                         % velocities (dx1,dx2)

% --- figure: scatter of my states ---
figure(4); clf; hold on; grid on; axis equal;
scatter(Xs(:,1), Xs(:,2), 18, 'filled', 'MarkerFaceAlpha', 0.55);
xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex');
title(sprintf('My data: states (subsampled %d/%d)', N_show, N_all), 'Interpreter','latex');
set(gca,'FontSize',12);

% --- figure: quiver of my point–velocity pairs (aligned) ---
% scale arrows to be visually comparable (robust to outliers)
mag = vecnorm(Vs,2,2);                        % speed
scl = max(prctile(mag,95), 1e-6);             % robust scale
U   = Vs(:,1) / scl; V = Vs(:,2) / scl;       % scale so typical arrows are ~1
figure(12); clf; hold on; grid on; axis equal;
quiver(Xs(:,1), Xs(:,2), U, V, 0, 'Color',[0.1 0.3 0.9]);  % 0 = no autoscale (we did it)
xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex');
title('My data: point–velocity pairs (quiver at sample locations)', 'Interpreter','latex');
set(gca,'FontSize',10);

% --- figure: overlay my points and my velocities (clean summary) ---
figure(5); clf; hold on; grid on; axis equal; box on;
scatter(Xs(:,1), Xs(:,2), 16, [0.85 0.6 0.05], 'filled', 'DisplayName','My samples');
quiver(Xs(:,1), Xs(:,2), U, V, 0, 'Color',[0 0.3 0.8], 'DisplayName','My velocities');
xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex');
%title('My dataset structure: $x, \dot x$ pairs', 'Interpreter','latex');
legend('Location','best'); set(gca,'FontSize',12);

% --- (optional) overlay colleague's field to show alignment visually ---
if exist('X_field','var') && exist('F_field','var')
    figure(6); clf; hold on; grid on; axis equal; box on;
    % colleague field (thin arrows so my samples pop)
    quiver(X_field(:,1), X_field(:,2), F_field(:,1), F_field(:,2), ...
           'AutoScale','on','Color',[0.2 0.2 0.6 0.35],'DisplayName','Their field');
    % my samples/arrows
    scatter(Xs(:,1), Xs(:,2), 14, [0.85 0.6 0.05], 'filled', 'DisplayName','My samples');
    quiver(Xs(:,1), Xs(:,2), U, V, 0, 'Color',[0 0.3 0.8], 'DisplayName','My velocities');
    xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex');
    %title('Comparison: their continuous field vs my (x,\dot x) samples', 'Interpreter','latex');
    %legend('Location','best'); set(gca,'FontSize',12);
end
