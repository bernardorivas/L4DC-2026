%% toggle_switch_to_GTdata.m
% Build toggle_switch_GT_data.mat from collaborator's toggle_switch_data.mat
% Uses ground-truth vector field and switching surfaces from SLS spec,
% consistent with construct.m: k = 1 + i + 2j, i=H(T1-x1), j=H(T2-x2), H(0)=1.

clear; clc; close all;

%% === 0) Load collaborator data (contains trajectory_data, F_field) ===
load toggle_switch_data.mat;
assert(exist('trajectory_data','var')==1 && exist('F_field','var')==1, ...
  'Expected variables `trajectory_data` and `F_field` inside toggle_switch_data.mat.');

X    = double(trajectory_data);   % N×2
Xdot = double(F_field);           % N×2
[N,n] = size(X);  assert(n==2,'Expected 2D state data.');

%% === 1) GT PARAMETERS (fill concrete values) ===
gamma1 = 1.0;
gamma2 = 1.0;
T1     = 3.0;
T2     = 3.0;
L1     = 1.0;
L2     = 1.0;
U1     = 5.0;
U2     = 5.0;

A = -diag([gamma1, gamma2]);     % common A for all four modes
tau = 0;                         % Heaviside tolerance (H(0)=1)

%% === 2) Switching surfaces and sign pairs (canonical: right/top are positive) ===
% f1 = x1 - T1, f2 = x2 - T2.  Treat ties as positive (>=0).
sgn_ge = @(u) 2*(u>=0) - 1;             % returns +1 for >=0, -1 for <0

f1 = X(:,1) - T1;                        % vertical surface
f2 = X(:,2) - T2;                        % horizontal surface
s1 = sgn_ge(f1);                         % + if x1 >= T1,  - if x1 < T1
s2 = sgn_ge(f2);                         % + if x2 >= T2,  - if x2 < T2

%% === 3) Map sign pairs → modes via q = 1 + i + 2 j ===
i_bit  = (s1 > 0);                       % 1 on right of T1
j_bit  = (s2 > 0);                       % 1 above T2
mode_gt = 1 + i_bit + 2*j_bit;           % {1,2,3,4} in BL,BR,TL,TR order

% If you had already built labels with the old convention,
% an equivalent quick fix is:  mode_gt = 5 - mode_gt_old;  % 180° rotation

%% === 4) Optional: verify GT vector field consistency ===
iH = double(X(:,1) <= T1);   % i = H(T1 - x1)
jH = double(X(:,2) <= T2);   % j = H(T2 - x2)
b1 = L1 + jH.*(U1 - L1);
b2 = L2 + iH.*(U2 - L2);
B  = [b1, b2];
Xdot_GT = (X * A.').' + B.';  Xdot_GT = Xdot_GT.';
res = Xdot - Xdot_GT;
fprintf('GT residual check: mean |res|_2 = %.3e, max |res|_2 = %.3e\n', ...
        mean(vecnorm(res,2,2)), max(vecnorm(res,2,2)));

%% === 5) Package under a different file name (does NOT overwrite your own data) ===
data = struct();
data.y    = X;
data.dy   = Xdot;
data.mode = mode_gt(:);
data.gt   = struct('A',A,'T',[T1,T2],'L',[L1,L2],'U',[U1,U2], ...
                   'surface_f1','T1 - x1','surface_f2','T2 - x2', ...
                   'mode_book',[-1 -1; +1 -1; -1 +1; +1 +1]);  % consistent with construct.m

save('toggle_switch_GT_data.mat','data');
fprintf('Saved toggle_switch_GT_data.mat with N=%d points (Q=4 labels).\n', N);

%% === 6) Optional quick visualization ===
cols = lines(4);
figure(11); clf; hold on; grid on; axis equal;
for q = 1:4
  scatter(X(mode_gt==q,1), X(mode_gt==q,2), 12, cols(q,:), 'filled');
end
xline(T1,'k--','LineWidth',1.0);
yline(T2,'k--','LineWidth',1.0);
legend({'Mode 1','Mode 2','Mode 3','Mode 4','x_1=T_1','x_2=T_2'},'Location','best');
xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex');
title('GT mode labels from switching surfaces','Interpreter','latex');
