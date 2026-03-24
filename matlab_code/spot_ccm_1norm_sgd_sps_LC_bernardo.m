%% sps_piecewise_vdp_from_colleague_GT.m
% Identification on colleague's piecewise Van der Pol dataset
% Ground truth guard: |x1| = 1
% Mode 1: |x1| < 1,  Mode 2: |x1| >= 1
% True dynamics (mu=0.5): x1' = x2; x2' = mu*(1-x1^2)*x2 - g(x1),
%   with g(x1)= x1^3 for |x1|<1 and g(x1)=x1 for |x1|>=1

clear; clc; close all;

%% ----------------- Load & auto-map variables -----------------
datafile = "piecewise_vdp_data.mat";
S = load(datafile);

% helper to pick first existing field name
pick = @(lst) S.(lst{find(isfield(S,lst),1,'first')});
try
    X    = pick({'trajectory_data','trajectory_points','X','x','states'});
    Xdot = pick({'F_field','vector_field_at_trajectory_data','Xdot','xdot','vel','f_at_x'});
catch
    error('Could not find required variables in %s. Available: %s', ...
        datafile, strjoin(fieldnames(S), ', '));
end
if size(X,2)~=2 || size(Xdot,2)~=2, error('Expecting Nx2 states/velocities.'); end
if size(X,1)~=size(Xdot,1), error('size mismatch: X vs Xdot.'); end

N = size(X,1);  n = 2;  Q = 2;
mu_true = 0.5;

%% ----------------- Ground-truth labels & VF -----------------
true_mode_all = ones(N,1);                    % mode 1 by default
true_mode_all(abs(X(:,1)) >= 1) = 2;          % mode 2 if |x1|>=1

Vgt_all = sps_true_rhs_vec(X, mu_true);       % analytic ground-truth f(x)

%% ----------------- Params -----------------
nIter = 30;                 % alternating iterations
d     = 3;                  % dynamics degree (cubic OK for VdP-type)
kappa = 2;                  % switching surface degree
rng(1);

%% ----------------- Normalize once -----------------
mx = mean(X,1); sx = std(X,[],1);  sx(sx==0)=1;
md = mean(Xdot,1); sd = std(Xdot,[],1); sd(sd==0)=1;
Xn    = (X - mx)./sx;              % normalized states
Xdotn = (Xdot - md)./sd;           % normalized velocities

% Features for dynamics
Phi_all_d = buildMonomialMatrix(Xn, d);
P = size(Phi_all_d, 2);

% exponent order (must match your buildMonomialMatrix)
exps_d = fliplr(generateExponentList(n, d));
exps_k = fliplr(generateExponentList(n, kappa));
if size(exps_d,1) ~= P, error('Mismatch Phi_all_d columns vs exps_d.'); end

%% ----------------- Batch / validation split -----------------
Nbatch  = min(2000, N);
id      = randperm(N, Nbatch);
xop     = Xn(id,:);                  % normalized states (train batch)
dz      = Xdotn(id,:);               % normalized velocities
raw_xop = X(id,:);                   % raw for guard distance
true_mode = true_mode_all(id,:);     % GT labels for batch

% ----- Validation set (GT labels from |x1|) -----
rng(123);
pool = setdiff(1:N, id);

if isempty(pool)
    % fall back: hold out a small subset from the batch itself
    Nval   = min(400, numel(id));
    id_val = id(randperm(numel(id), Nval));
else
    Nval   = min(400, numel(pool));
    id_val = pool(randperm(numel(pool), Nval));
end

Xval    = Xn(id_val,:);                      % normalized for features
dz_valn = Xdotn(id_val,:);                   % normalized velocities
gt_val  = 1 + (abs(X(id_val,1)) >= 1);       % GT guard: |x1|<1 -> 1, else 2

val_error = nan(nIter,1);                    % init (NaN if we skip)

% Confident subset (away from the true guard |x1|=1 in RAW)
band      = 0.25;
confmask  = abs(abs(raw_xop(:,1)) - 1) > band;
id_conf   = find(confmask);
id_all    = (1:Nbatch)';

%% ----------------- Init (unsupervised) -----------------
[~, ~, flat] = init_poly_coeffs_unsup(xop, dz, d, Q, ...
    'Rounds', 5, 'Ridge', 1e-6, 'Tau', 1e-2, 'Seed', 'global+noise');
a1 = flat.a1; b1 = flat.b1; a2 = flat.a2; b2 = flat.b2;

%% ----------------- Alternating scheme -----------------
cost_poly_log = zeros(nIter,1);
mode_error    = zeros(nIter,1);
% val_error     = zeros(nIter,1);

for iter = 1:nIter
    use_conf_only = (iter <= 4);
    if use_conf_only, idx_fit = id_conf; else, idx_fit = id_all; end

    % Features for batch/fit sets
    Phi_batch = Phi_all_d(id,:);          % (Nbatch x P)
    Phi_fit   = Phi_batch(idx_fit,:);     % (|fit| x P)
    dz_fit    = dz(idx_fit,:);
    raw_fit   = raw_xop(idx_fit,:);

    % Per-mode predictions (normalized)
    H1 = [Phi_fit*a1, Phi_fit*b1];
    H2 = [Phi_fit*a2, Phi_fit*b2];

    % Robust balancing (componentwise MAD)
    dz_med   = median(dz_fit, 1);
    dz_mad   = mad(dz_fit, 1, 1);  dz_mad(dz_mad < 1e-8) = 1;
    dz_fit_n = (dz_fit - dz_med) ./ dz_mad;
    H1_n     = (H1 - dz_med) ./ dz_mad;
    H2_n     = (H2 - dz_med) ./ dz_mad;

    % Weights: taper near guard (in RAW, true guard is |x1|=1)
    band_w   = 0.30;
    dist_raw = abs(abs(raw_fit(:,1)) - 1);
    w_fit    = min(1, dist_raw / band_w);

    %% Step A: mode LP (weighted L1 on fit set)
    prog_mode = spotsosprog;
    [sol_mode, ~, ~, vars_mode] = searchMode_weighted(dz_fit_n, {H1_n, H2_n}, prog_mode, w_fit);
    lam_fit = double(sol_mode.eval(vars_mode.lam));   % |fit| x 2

    % Lift to full batch; early sharpening
    lambda_soft = (1/Q)*ones(length(id_all),Q);
    lambda_soft(idx_fit,:) = lam_fit;
    if iter <= 4
        gamma = 2.0;
        lambda_soft = lambda_soft.^gamma;
        lambda_soft = lambda_soft ./ sum(lambda_soft,2);
    end
    [~, mode_op] = max(lambda_soft, [], 2);

    % Resolve permutation vs GT labels
    if ~isempty(true_mode)
        m1 = nnz(mode_op ~= true_mode);
        m2 = nnz((3 - mode_op) ~= true_mode);
        if m2 < m1
            mode_op     = 3 - mode_op;
            lambda_soft = lambda_soft(:,[2 1]);
        end
        mode_error(iter) = nnz(mode_op ~= true_mode);
    end

    %% Step B: poly fit (soft λ for first 6 iters)
    use_soft = (iter <= 6);
    if use_soft
        lambda_used = lambda_soft;
    else
        lambda_used = sparse(1:length(mode_op), mode_op, 1, length(mode_op), Q);
        lambda_used = full(lambda_used);
    end

    prog_poly = spotsosprog;
    if use_conf_only
        [sol_poly, prog_poly, ~, vars_poly] = searchPoly(xop(idx_fit,:), dz(idx_fit,:), lambda_used(idx_fit,:), prog_poly, Phi_fit);
    else
        [sol_poly, prog_poly, ~, vars_poly] = searchPoly(xop,  dz,    lambda_used,                prog_poly, Phi_batch);
    end

    a1 = double(sol_poly.eval(vars_poly.a1));
    b1 = double(sol_poly.eval(vars_poly.b1));
    a2 = double(sol_poly.eval(vars_poly.a2));
    b2 = double(sol_poly.eval(vars_poly.b2));

    cost_poly_log(iter) = double(sol_poly.eval(sum(vars_poly.delta) * 1e3));
    
    % ----- Validation (only if we actually have a val set) -----
    if Nval > 0
        Phi_val = buildMonomialMatrix(Xval, d);
        H1v = [Phi_val*a1, Phi_val*b1];
        H2v = [Phi_val*a2, Phi_val*b2];
        prog_val = spotsosprog;
        [sol_val, ~, ~, vars_val] = searchMode(dz_valn, {H1v, H2v}, prog_val);
        lam_val  = double(sol_val.eval(vars_val.lam));
        [~, mode_val] = max(lam_val, [], 2);
        val_error(iter) = nnz(mode_val ~= gt_val);
        fprintf('Iter %2d | PolyID loss=%.3e | Train mismatch=%d | Val mismatch=%d\n', ...
            iter, cost_poly_log(iter), mode_error(iter), val_error(iter));
    else
        fprintf('Iter %2d | PolyID loss=%.3e | Train mismatch=%d | Val: (skipped)\n', ...
            iter, cost_poly_log(iter), mode_error(iter));
    end
end

%% ----------------- Learn switching surface -----------------
eta=10; epsilon=1e-2; beta=1e-2;
sigma = 2*(mode_op==1) - 1;
[a_opt, z_val, diagnostics] = searchSwitchingSurface_softmargin(xop, sigma, kappa, eta, epsilon, beta);

%% ----------------- Diagnostics vs TRUE vector field -----------------
% Compare on the batch points (RAW)
Xb   = X(id,:);
Vgt  = sps_true_rhs_vec(Xb, mu_true);
V_or = sps_learned_rhs_oracle(     Xb, a1,b1,a2,b2, d, mx,sx, md,sd);            % guarded by true |x1|
V_le = sps_learned_rhs_learnedguard(Xb, a1,b1,a2,b2, d, a_opt, kappa, mx,sx, md,sd); % guarded by learned f

MAE_or = mean(abs(V_or - Vgt),1); MAE_le = mean(abs(V_le - Vgt),1);
ANG_or = mean(ang_err_deg(V_or, Vgt)); ANG_le = mean(ang_err_deg(V_le, Vgt));
fprintf('\nBatch stats vs TRUE | MAE(true-OR)=%.2e,%.2e | MAE(true-LE)=%.2e,%.2e | ANG=%.2f/%.2f deg\n', ...
    MAE_or(1),MAE_or(2), MAE_le(1),MAE_le(2), ANG_or, ANG_le);

train_acc = 1 - mode_error(end)/numel(id);
if Nval > 0
    val_acc = 1 - val_error(end)/numel(gt_val);
    fprintf('Mode accuracy | Train: %.2f%% | Val: %.2f%%\n', 100*train_acc, 100*val_acc);
else
    fprintf('Mode accuracy | Train: %.2f%% | Val: (no holdout)\n', 100*train_acc);
end


%% ----------------- Plots -----------------
set(groot, 'defaultFigureColor', 'w', ...
           'defaultTextInterpreter','latex', ...
           'defaultAxesTickLabelInterpreter','latex', ...
           'defaultLegendInterpreter','latex');

figure(1); clf; hold on; grid on; box on;
plot(abs(cost_poly_log)+1e-18, '-o', 'LineWidth', 1.2);
ylabel('Poly ID loss'); xlabel('Iter'); set(gca,'YScale','log');
title('Convergence');

% Surface vs points (raw)
xop_raw = X(id,:);
idx1 = (mode_op==1); idx2 = ~idx1;
[x1r,x2r] = meshgrid(linspace(min(X(:,1)),max(X(:,1)),300), linspace(min(X(:,2)),max(X(:,2)),300));
Xgrid_raw = [x1r(:), x2r(:)];
Xgrid_n   = (Xgrid_raw - mx)./sx;
Phi_grid  = buildMonomialMatrix(Xgrid_n, kappa);
Z         = reshape(Phi_grid*a_opt, size(x1r));
figure(2); clf; hold on; axis equal; grid on; box on;
h1=scatter(xop_raw(idx1,1),xop_raw(idx1,2),10,[.9 .7 0],'filled','DisplayName','Mode 1');
h2=scatter(xop_raw(idx2,1),xop_raw(idx2,2),10,[0 0 .8],'filled','DisplayName','Mode 2');
[~,h3]=contour(x1r,x2r,Z,[0 0],'k','LineWidth',2); h3.DisplayName='$f(x)=0$ (learned)';
%xline(1,'r--','LineWidth',1); xline(-1,'r--','LineWidth',1); % true guard
%legend([h1 h2 h3],'Location','best');
legend([h1 h2], 'Location', 'best');
xlabel('$x_1$'); ylabel('$x_2$');
%title('Identified surface vs true guard (red)');

% Quiver: data vs learned (guarded by learned f)
Ns = min(400, N);
sub = randperm(N, Ns);
Xn_sub   = (X(sub,:) - mx)./sx;
Phi_sub  = buildMonomialMatrix(Xn_sub, d);
V1n = [Phi_sub*a1, Phi_sub*b1];
V2n = [Phi_sub*a2, Phi_sub*b2];
Phi_k_sub = buildMonomialMatrix(Xn_sub, kappa);
f_sub = Phi_k_sub * a_opt;
Vn   = V2n;  Vn(f_sub >= 0,:) = V1n(f_sub >= 0,:);
Vraw = Vn.*sd + md;

figure(3); clf; hold on; axis equal; grid on; box on;
scatter(X(sub,1), X(sub,2), 8, 'k', 'filled', 'DisplayName','Trajectory points');
scale = 0.15;
quiver(X(sub,1), X(sub,2), Xdot(sub,1)*scale, Xdot(sub,2)*scale, 0, 'b', 'DisplayName','Data f(x)');
quiver(X(sub,1), X(sub,2), Vraw(:,1)*scale, Vraw(:,2)*scale, 0, 'r', 'DisplayName','Learned f(x)');
legend('Location','best'); xlabel('$x_1$'); ylabel('$x_2$');
title('Vector field: data vs learned');

%% ----------------- EXPORT (normalized + RAW) -----------------
assert(size(Phi_all_d,2)==numel(a1) && numel(a1)==numel(b1) && numel(a1)==numel(a2) && numel(a1)==numel(b2));
if ~isempty(exps_d), assert(size(exps_d,1)==numel(a1)); end
if ~isempty(exps_k), assert(size(exps_k,1)==numel(a_opt)); end

% Evaluate surface & hard labels on all data
Xn_all    = (X - mx) ./ sx;
Phi_all_k = buildMonomialMatrix(Xn_all, kappa);
fX        = Phi_all_k * a_opt;
mode_f    = ones(size(fX)); mode_f(fX < 0) = 2;
sigma_f   = 2*(mode_f==1) - 1;

% Per-mode dynamics on all data (norm → RAW)
Phi_all_d = buildMonomialMatrix(Xn_all, d);
VH1n = [Phi_all_d*a1, Phi_all_d*b1];
VH2n = [Phi_all_d*a2, Phi_all_d*b2];
VH1  = VH1n .* sd + md;   % RAW
VH2  = VH2n .* sd + md;
C_all = [sum((Xdot - VH1).^2,2), sum((Xdot - VH2).^2,2)];
[~, mode_resid] = min(C_all, [], 2);

% Model struct
model = struct();
model.type='SPS'; model.n=2; model.Q=2;
model.degree_d=d; model.surface_deg=kappa;
model.norm.mx=mx(:)'; model.norm.sx=sx(:)';
model.norm.md=md(:)'; model.norm.sd=sd(:)';
model.basis_d.exponents = exps_d;
model.surface.exponents = exps_k;
model.modes(1).name='mode1'; model.modes(1).dx1_coeffs=a1(:); model.modes(1).dx2_coeffs=b1(:);
model.modes(2).name='mode2'; model.modes(2).dx1_coeffs=a2(:); model.modes(2).dx2_coeffs=b2(:);
model.surface.coeffs=a_opt(:);
model.surface.description='f(x_norm)>=0 -> mode1; f(x_norm)<0 -> mode2; x_norm=(x-mx)./sx';
model.full.X = X; model.full.Xdot = Xdot;
model.full.fX = fX; model.full.mode_fsign = mode_f;
model.full.mode_resid = mode_resid;
model.metadata.created = datestr(now,'yyyy-mm-ddTHH:MM:SS');
model.metadata.description = 'Bilevel SPS ID on piecewise-VdP with GT labels & VF.';

save('sps_identified_model.mat','model');
fid=fopen('sps_identified_model.json','w'); fwrite(fid,jsonencode(model)); fclose(fid);

% CSV: per-mode coeffs (normalized basis)
mode_col=[]; comp_col=[]; e1_col=[]; e2_col=[]; coef_col=[];
append_block = @(mode,coeffs,tag) deal( ...
    [mode_col; mode*ones(size(coeffs,1),1)], ...
    [comp_col; repmat(string(tag), size(coeffs,1),1)], ...
    [e1_col;   exps_d(:,1)], ...
    [e2_col;   exps_d(:,2)], ...
    [coef_col; coeffs(:)] );
[mode_col,comp_col,e1_col,e2_col,coef_col] = append_block(1,a1,'dx1');
[mode_col,comp_col,e1_col,e2_col,coef_col] = append_block(1,b1,'dx2');
[mode_col,comp_col,e1_col,e2_col,coef_col] = append_block(2,a2,'dx1');
[mode_col,comp_col,e1_col,e2_col,coef_col] = append_block(2,b2,'dx2');
writetable(table(mode_col,comp_col,e1_col,e2_col,coef_col, ...
    'VariableNames',{'mode','component','e1','e2','coeff'}), 'sps_mode_polynomials.csv');

if ~isempty(exps_k)
    writetable(table(exps_k(:,1), exps_k(:,2), a_opt(:), ...
        'VariableNames',{'e1','e2','a'}), 'sps_switch_surface.csv');
end

writetable(table(X(:,1),X(:,2),Xdot(:,1),Xdot(:,2),fX,mode_f,sigma_f,true_mode_all, ...
    'VariableNames',{'x1','x2','x1dot','x2dot','fX','mode_fsign','sigma','mode_true'}), ...
    'sps_full_data_with_modes.csv');

% RAW expansions
surface_raw_tbl = local_expand_surface_to_raw(a_opt, exps_k, mx, sx);
writetable(surface_raw_tbl, 'sps_switch_surface_RAW.csv');

T_all = table();
T = local_expand_dyn_to_raw(a1(:), exps_d, mx, sx, sd(1), md(1)); T.mode(:)=1; T.component(:)="dx1"; T_all=[T_all; T];
T = local_expand_dyn_to_raw(b1(:), exps_d, mx, sx, sd(2), md(2)); T.mode(:)=1; T.component(:)="dx2"; T_all=[T_all; T];
T = local_expand_dyn_to_raw(a2(:), exps_d, mx, sx, sd(1), md(1)); T.mode(:)=2; T.component(:)="dx1"; T_all=[T_all; T];
T = local_expand_dyn_to_raw(b2(:), exps_d, mx, sx, sd(2), md(2)); T.mode(:)=2; T.component(:)="dx2"; T_all=[T_all; T];
T_all = movevars(T_all, {'mode','component'}, 'Before', 1);
writetable(T_all, 'sps_mode_polynomials_RAW.csv');

disp('Export complete: .mat/.json/.csv (normalized + RAW).');

%% ===== Identified switching system: regions + learned VF (Mode 1/2) =====
% domain (raw units) with a little padding
pad  = 0.10*(max(X)-min(X));
xbox = [min(X(:,1))-pad(1), max(X(:,1))+pad(1), ...
        min(X(:,2))-pad(2), max(X(:,2))+pad(2)];

GRID_N   = 220;         % background resolution
QUIVER_N = 24;          % quiver grid per axis
AR_SCALE = 3.0;

x1r = linspace(xbox(1), xbox(2), GRID_N);
x2r = linspace(xbox(3), xbox(4), GRID_N);
[XX,YY] = meshgrid(x1r, x2r);

% --- evaluate learned guard f on grid (guard is defined in normalized coords) ---
Xg_n  = ([XX(:), YY(:)] - mx)./sx;
Phi_g = buildMonomialMatrix(Xg_n, kappa);
fg    = reshape(Phi_g * a_opt, size(XX));

% --- region labels from learned guard: Mode 1 if f >= 0, Mode 2 otherwise ---
mode_grid       = ones(size(fg));
mode_grid(fg<0) = 2;

% --- learned vector field on a coarser quiver grid (raw units) ---
x1q = linspace(xbox(1), xbox(2), QUIVER_N);
x2q = linspace(xbox(3), xbox(4), QUIVER_N);
[XQ,YQ] = meshgrid(x1q, x2q);

Xq_n  = ([XQ(:), YQ(:)] - mx)./sx;
Phi_q = buildMonomialMatrix(Xq_n, d);
V1n_q = [Phi_q*a1, Phi_q*b1];
V2n_q = [Phi_q*a2, Phi_q*b2];

Phi_k_q = buildMonomialMatrix(Xq_n, kappa);
fq      = Phi_k_q * a_opt;

Vn_q          = V2n_q;                 % default mode 2
Vn_q(fq>=0,:) = V1n_q(fq>=0,:);        % mode 1 where f>=0
Vraw_q        = Vn_q.*sd + md;
U = reshape(Vraw_q(:,1), size(XQ));
V = reshape(Vraw_q(:,2), size(YQ));

%% === Plotting (discrete colors, export-safe, no rim gaps / seams) ===
figure(4); clf; hold on; box on; axis equal;

% background regions
hImg = imagesc(x1r, x2r, mode_grid);
set(gca,'YDir','normal'); grid on;

% expand image to pixel edges → removes white rim with axis equal
dx = x1r(2)-x1r(1);
dy = x2r(2)-x2r(1);
set(hImg, 'XData', [x1r(1)-dx/2, x1r(end)+dx/2], ...
          'YData', [x2r(1)-dy/2, x2r(end)+dy/2]);

axis image;
xlim([x1r(1)-dx/2, x1r(end)+dx/2]);
ylim([x2r(1)-dy/2, x2r(end)+dy/2]);

% two discrete colors (Mode 1 blue, Mode 2 orange)
cols_all = lines(4);
cols2    = [cols_all(1,:); cols_all(2,:)];
colormap(cols2);
clim([0.5 2.5]);     % freeze discrete bins {1,2}

% overlays
contour(XX,YY,fg,[0 0],'k','LineWidth',1.4);   % learned f=0
quiver(XQ, YQ, U, V, AR_SCALE, 'Color',[0 0 0]);

xlabel('$x_1$','Interpreter','latex');
ylabel('$x_2$','Interpreter','latex');

% dot-style legend (no colorbar)
hL = [scatter(nan,nan,30,cols2(1,:),'filled'), ...
      scatter(nan,nan,30,cols2(2,:),'filled')];
legend(hL, {'Mode 1','Mode 2'}, 'Location','best', ...
       'Interpreter','latex','Box','on');

% renderer/export (image rasterized → no PDF seam; lines stay vector)
set(gcf,'Renderer','opengl','Color','w');
% exportgraphics(gcf,'sps_identified_vf_regions.pdf','ContentType','vector','BackgroundColor','white');
% exportgraphics(gcf,'sps_identified_vf_regions.png','Resolution',400);


%% ======================= Helpers =======================
function [sol, prog, cost_vec, vars] = searchMode_weighted(dz, dz_tilde, prog_base, w)
    % Weighted L1 simplex LP
    N = size(dz,1); M = numel(dz_tilde); n = size(dz,2);
    [prog, lam] = prog_base.newPos(N*M); 
    lam = reshape(lam,[N,M]);

    cost_vec = [];
    for k=1:N
        dzk = 0;
        for q=1:M, dzk = dzk + dz_tilde{q}(k,:)' * lam(k,q); end
        cost_vec = [cost_vec; dz(k,:)' - dzk];
    end

    [prog, t] = prog.newPos(numel(cost_vec));
    prog = prog.withPos(t - cost_vec);
    prog = prog.withPos(t + cost_vec);

    prog = prog.withPos(1 - lam(:)); prog = prog.withPos(lam(:));
    for k=1:N, prog = prog.withEqs(1 - sum(lam(k,:))); end

    Wt = kron(w(:), ones(n,1)); Wt = Wt(:);
    obj = 0; for i=1:numel(t), obj = obj + Wt(i)*t(i); end

    options = spot_sdp_default_options(); options.verbose = 0;
    sol = prog.minimize(obj * 1e3, @spot_mosek, options);

    vars.lam = lam; vars.t = t;
    cost_vec = t; % for completeness
end

function V = sps_true_rhs_vec(Xraw, mu_)
    x1 = Xraw(:,1); x2 = Xraw(:,2);
    xterm = x1;
    mask = abs(x1) < 1; xterm(mask) = x1(mask).^3;
    V = [ x2, mu_*(1 - x1.^2).*x2 - xterm ];
end

function Vraw = sps_learned_rhs_oracle(Xraw, a1,b1,a2,b2,deg, mx,sx, md,sd)
    % Uses TRUE guard |x1|=1 to pick modes
    Xn = (Xraw - mx)./sx;
    Phi = buildMonomialMatrix(Xn,deg);
    V1n = [Phi*a1, Phi*b1]; V2n = [Phi*a2, Phi*b2];
    Vn  = V2n; Vn(abs(Xraw(:,1)) < 1,:) = V1n(abs(Xraw(:,1)) < 1,:);
    Vraw = Vn.*sd + md;
end

function Vraw = sps_learned_rhs_learnedguard(Xraw, a1,b1,a2,b2,deg, a_opt, kappa, mx,sx, md,sd)
    Xn = (Xraw - mx)./sx;
    Phi_k = buildMonomialMatrix(Xn,kappa);
    fn = Phi_k * a_opt;
    Phi = buildMonomialMatrix(Xn,deg);
    V1n = [Phi*a1, Phi*b1]; V2n = [Phi*a2, Phi*b2];
    Vn  = V2n; Vn(fn >= 0,:) = V1n(fn >= 0,:);   % learned guard
    Vraw = Vn.*sd + md;
end

function ang = ang_err_deg(Vhat, Vtrue)
    epsn = 1e-12;
    c = sum(Vhat.*Vtrue,2) ./ max(vecnorm(Vhat,2,2).*vecnorm(Vtrue,2,2), epsn);
    c = max(-1,min(1,c));
    ang = acosd(c);
end

function T = local_expand_surface_to_raw(a_opt, exps_k, mx, sx)
    if isempty(exps_k)
        T = table([],[],[], 'VariableNames',{'e1','e2','a_raw'}); return;
    end
    inv_sx1 = 1.0 / sx(1); inv_sx2 = 1.0 / sx(2);
    Araw = containers.Map('KeyType','char','ValueType','double');
    e1k = exps_k(:,1); e2k = exps_k(:,2);

    for t = 1:numel(a_opt)
        i = e1k(t);  j = e2k(t);  aij = a_opt(t);
        for p = 0:i
            bin1 = nchoosek(i,p) * ((-mx(1))^(i-p)) * (inv_sx1^i);
            for q = 0:j
                bin2 = nchoosek(j,q) * ((-mx(2))^(j-q)) * (inv_sx2^j);
                key = sprintf('%d_%d', p, q);
                if isKey(Araw, key), prev = Araw(key); else, prev = 0.0; end
                Araw(key) = prev + aij * bin1 * bin2;
            end
        end
    end

    keysCell = Araw.keys();
    e1r=[]; e2r=[]; ar=[];
    for kk = 1:numel(keysCell)
        pq = sscanf(keysCell{kk}, '%d_%d');
        e1r(end+1,1) = pq(1);
        e2r(end+1,1) = pq(2);
        ar(end+1,1)  = Araw(keysCell{kk});
    end
    T = sortrows(table(e1r,e2r,ar,'VariableNames',{'e1','e2','a_raw'}), {'e1','e2'});
end

function T = local_expand_dyn_to_raw(coeffs, exps_d, mx, sx, out_gain, out_bias)
    if isempty(exps_d), error('exps_d required to label RAW coefficients.'); end
    inv_sx1 = 1.0 / sx(1); inv_sx2 = 1.0 / sx(2);
    Araw = containers.Map('KeyType','char','ValueType','double');

    for t = 1:numel(coeffs)
        c_ij = coeffs(t);
        if abs(c_ij) < 1e-18, continue; end
        i = exps_d(t,1);  j = exps_d(t,2);
        for p = 0:i
            bin1 = nchoosek(i,p) * ((-mx(1))^(i-p)) * (inv_sx1^i);
            for q = 0:j
                bin2 = nchoosek(j,q) * ((-mx(2))^(j-q)) * (inv_sx2^j);
                key = sprintf('%d_%d', p, q);
                if isKey(Araw, key), prev = Araw(key); else, prev = 0.0; end
                Araw(key) = prev + out_gain * c_ij * bin1 * bin2;
            end
        end
    end
    key00 = '0_0';
    if isKey(Araw, key00), prev = Araw(key00); else, prev = 0.0; end
    Araw(key00) = prev + out_bias;

    keysCell = Araw.keys();
    e1r=[]; e2r=[]; cr=[];
    for kk = 1:numel(keysCell)
        pq = sscanf(keysCell{kk}, '%d_%d');
        e1r(end+1,1) = pq(1);
        e2r(end+1,1) = pq(2);
        cr(end+1,1)  = Araw(keysCell{kk});
    end
    T = sortrows(table(e1r,e2r,cr,'VariableNames',{'e1','e2','coeff_raw'}), {'e1','e2'});
end
