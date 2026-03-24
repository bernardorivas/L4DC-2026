%% plot_surfaces_and_vf_from_csv.m
% Plots: f1=0, f2=0 (from sls4_surfaces_coeffs_long.csv)
%        identified VF (from sls4_vf_coeffs.csv) over 4 regions.
%
% Modes 1..4 correspond to sign pairs:
%   1: (-,-), 2: (+,-), 3: (-,+), 4: (+,+)

clear; clc; close all;

%% ---------------- Files ----------------
surfaces_csv = 'sls4_surfaces_coeffs_long.csv';
vf_csv       = 'sls4_vf_coeffs.csv';
mat_optional = 'sls4_two_surface_model.mat';   % optional for data-range only

assert(exist(surfaces_csv,'file')==2, 'Missing %s', surfaces_csv);
assert(exist(vf_csv,'file')==2,       'Missing %s', vf_csv);

%% ---------------- Load surfaces (long format) ----------------
Ts = readtable(surfaces_csv);
T1 = Ts(strcmp(Ts.surface,'f1'), :);
T2 = Ts(strcmp(Ts.surface,'f2'), :);
assert(~isempty(T1) && ~isempty(T2), 'CSV must contain rows for f1 and f2');

E1 = [T1.e1, T1.e2, T1.a];
E2 = [T2.e1, T2.e2, T2.a];

%% ---------------- Load vector field from CSV ----------------
Tv = readtable(vf_csv);
mmax = max(Tv.mode);
assert(mmax==4, 'This helper expects Q=4');

n = max(Tv.row(strcmp(Tv.type,'A')));   % infer n from A rows
A = zeros(n,n,mmax);
b = zeros(n,mmax);
for q = 1:mmax
    selA = Tv.mode==q & strcmp(Tv.type,'A');
    rA = Tv.row(selA); cA = Tv.col(selA); vA = Tv.value(selA);
    for k = 1:numel(vA), A(rA(k), cA(k), q) = vA(k); end
    selb = Tv.mode==q & strcmp(Tv.type,'b');
    rb = Tv.row(selb); vb = Tv.value(selb);
    for k = 1:numel(vb), b(rb(k), q) = vb(k); end
end

%% ---------------- Plotting domain ----------------
GRID_N   = 150;
QUIVER_N = 23;
ARROW_SCALE = 1.0;

xbox = [];
if exist(mat_optional,'file')==2
    S = load(mat_optional);
    if isfield(S,'model2') && isfield(S.model2,'full') && isfield(S.model2.full,'X')
        X = S.model2.full.X;
        pad = 0.15*(max(X)-min(X));
        xbox = [min(X(:,1))-pad(1), max(X(:,1))+pad(1), ...
                min(X(:,2))-pad(2), max(X(:,2))+pad(2)];
    end
end
if isempty(xbox), xbox = [0, 6, 0, 6]; end

x1r = linspace(xbox(1), xbox(2), GRID_N);
x2r = linspace(xbox(3), xbox(4), GRID_N);
[XX,YY] = meshgrid(x1r,x2r);

%% ---------------- Evaluate surfaces on grid ----------------
Z1 = eval_poly_on_grid(E1, XX, YY);   % f1(x)
Z2 = eval_poly_on_grid(E2, XX, YY);   % f2(x)

% Signs (no flips; we only relabel the colored regions below)
S1 = ones(size(Z1)); S1(Z1<0) = -1;
S2 = ones(size(Z2)); S2(Z2<0) = -1;

% Canonical sign→mode book
Sbook = [-1 -1; +1 -1; -1 +1; +1 +1];

% Assign each grid cell to a mode by sign pair
mode_grid = zeros(size(Z1));
for k = 1:4
    mode_grid( S1==Sbook(k,1) & S2==Sbook(k,2) ) = k;
end
if any(mode_grid(:)==0)
    mode_grid = fill_zeros_by_residual(mode_grid, XX, YY, A, b);
end

% ---- FIX: recolor only (swap top/bottom labels), keep arrows unchanged ----
% vertical flip of labels: 1<->3, 2<->4
M = mode_grid;
mode_grid = zeros(size(M));
mode_grid(M==1) = 3;
mode_grid(M==2) = 4;
mode_grid(M==3) = 1;
mode_grid(M==4) = 2;

%% ---------------- Figure A: Surfaces only ----------------
figure(1); clf; hold on; box on; grid on; axis equal;
contour(XX,YY,Z1,[0 0],'k','LineWidth',1.8);
contour(XX,YY,Z2,[0 0],'k','LineWidth',1.8);
xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex');
title('Zero level sets:  $f_1(x)=0$,  $f_2(x)=0$','Interpreter','latex');
legend({'$f_1=0$','$f_2=0$'},'Location','best','Interpreter','latex');

%% ---------------- Figure B: VF + regions ----------------
x1q = linspace(xbox(1), xbox(2), QUIVER_N);
x2q = linspace(xbox(3), xbox(4), QUIVER_N);
[XQ,YQ] = meshgrid(x1q, x2q);

% Decide mode at quiver points from surfaces (unchanged)
F1q = eval_poly_on_grid(E1, XQ, YQ);
F2q = eval_poly_on_grid(E2, XQ, YQ);
S1q = ones(size(F1q)); S1q(F1q<0) = -1;
S2q = ones(size(F2q)); S2q(F2q<0) = -1;
mode_q = zeros(size(F1q));
for k = 1:4
    mode_q(S1q==Sbook(k,1) & S2q==Sbook(k,2)) = k;
end
mode_q(mode_q==0) = 1;

% Compute arrows from CSV A,b (unchanged)
U = zeros(size(XQ)); V = zeros(size(YQ));
for k = 1:numel(XQ)
    m = mode_q(k);
    x = [XQ(k); YQ(k)];
    v = A(:,:,m)*x + b(:,m);
    U(k) = v(1);  V(k) = v(2);
end

figure(2); clf; hold on; box on; grid on; axis equal;

% --- discrete region image ---
hImg = imagesc(x1r, x2r, mode_grid);
set(gca,'YDir','normal');
cmap = lines(4);
colormap(cmap);
clim([0.5 4.5]);   % lock to integer color bins

% --- overlays ---
contour(XX,YY,Z1,[0 0],'k','LineWidth',1.2);
contour(XX,YY,Z2,[0 0],'k','LineWidth',1.2);
quiver(XQ, YQ, U, V, ARROW_SCALE, 'Color',[0 0 0]);

xlabel('$x_1$', 'Interpreter','latex');
ylabel('$x_2$', 'Interpreter','latex');
%title('Identified VF with switching surfaces','Interpreter','latex');

% === dot-style legend instead of colorbar ===
cols = lines(4);
hL = gobjects(4,1);
for q = 1:4
    % invisible tiny scatter handles for legend
    hL(q) = scatter(nan, nan, 25, cols(q,:), 'filled');
end
legend(hL, {'Mode 1','Mode 2','Mode 3','Mode 4'}, ...
       'Location','best','Interpreter','latex','Box','on');

% (optional) for consistent export colors
set(gcf,'Renderer','painters');
exportgraphics(gcf,'fig_vf_id_surface_example1.pdf','ContentType','vector');
exportgraphics(gcf,'fig_vf_id_surface_example1.png');

%% ---------------- Helpers ----------------
function Z = eval_poly_on_grid(E, XX, YY)
Z = zeros(size(XX));
for k = 1:size(E,1)
    e1 = E(k,1); e2 = E(k,2); a = E(k,3);
    if a~=0, Z = Z + a .* (XX.^e1) .* (YY.^e2); end
end
end

function mode_grid = fill_zeros_by_residual(mode_grid, XX, YY, A, b)
[idx_i, idx_j] = find(mode_grid==0);
Q = size(A,3);
for t = 1:numel(idx_i)
    i = idx_i(t); j = idx_j(t);
    x = [XX(i,j); YY(i,j)];
    best = 1; bestR = inf;
    for q = 1:Q
        r = norm(A(:,:,q)*x + b(:,q), 1);
        if r < bestR, best = q; bestR = r; end
    end
    mode_grid(i,j) = best;
end
end
