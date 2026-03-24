%% spot_ccm_1norm_sgd_sps_LC.m  (cleaned + fixed CSV export)
% System:
%   x' = [x2; mu*(1-x1^2)*x2 - x1^3]  if |x1|<1
%        [x2; mu*(1-x1^2)*x2 - x1]    if |x1|>=1

clear; clc; close all;

%% ----------------- Load data -----------------
load("sps_limit_cycle_data.mat");   % data.y, data.dy, data.mode, mu, hyst
X    = data.y;                      % (N x 2)
Xdot = data.dy;                     % (N x 2)
N    = size(X, 1);  n = 2;  Q = 2;

%% ----------------- Params -----------------
nIter = 30;                 % alternating iterations
d     = 3;                  % dynamics degree (needs cubic)
kappa = 2;                  % switching surface degree
rng(1);

%% ----------------- Normalize once -----------------
mx = mean(X,1); sx = std(X,[],1);  sx(sx==0)=1;
md = mean(Xdot,1); sd = std(Xdot,[],1); sd(sd==0)=1;
Xn    = (X - mx)./sx;              % normalized states
Xdotn = (Xdot - md)./sd;           % normalized velocities

% Build monomial feature matrix for dynamics
Phi_all_d = buildMonomialMatrix(Xn, d);
P = size(Phi_all_d, 2);

% Your buildMonomialMatrix uses: exp_list = generateExponentList(...); exp_list = fliplr(exp_list);
exps_d = fliplr(generateExponentList(n, d));   % exponent order for dynamics
exps_k = fliplr(generateExponentList(n, kappa));  % exponent order for surface

% sanity check
if size(exps_d,1) ~= P
    error('Mismatch between Phi_all_d columns and number of exponent rows.');
end


%% ----------------- Batch split -----------------
Nbatch  = min(2000, N);
id      = randperm(N, Nbatch);
xop     = Xn(id,:);                  % norm states (train batch)
dz      = Xdotn(id,:);               % norm velocities
raw_xop = X(id,:);                   % raw states (for guard distance)
true_mode = data.mode(id,:);         % ground-truth labels if available

% Fixed validation set (RAW guard labels for stability)
rng(123);
pool    = setdiff(1:N, id);
Nval    = min(400, numel(pool));
id_val  = pool(randperm(numel(pool), Nval));
Xval    = X(id_val,:);               % RAW states for guard
dz_valn = Xdotn(id_val,:);           % norm velocities
gt_val  = 1 + (abs(Xval(:,1)) > 1);  % raw guard labels

% Confident subset (away from |x1|=1)
band      = 0.25;                    % RAW units
confmask  = abs(abs(raw_xop(:,1)) - 1) > band;
id_conf   = find(confmask);
id_all    = (1:Nbatch)';

%% ----------------- Init (unsupervised) -----------------
% Your initializer should return coeffs in the same monomial order as Phi_all_d
[coeffs0, lambda0, flat] = init_poly_coeffs_unsup(xop, dz, d, Q, ...
    'Rounds', 5, 'Ridge', 1e-6, 'Tau', 1e-2, 'Seed', 'global+noise');
a1 = flat.a1; b1 = flat.b1; a2 = flat.a2; b2 = flat.b2;

%% ----------------- Alternating scheme -----------------
cost_poly_log = zeros(nIter,1);
mode_error    = zeros(nIter,1);

for iter = 1:nIter
    use_conf_only = (iter <= 4);
    if use_conf_only
        idx_fit = id_conf;
    else
        idx_fit = id_all;
    end


    % Features for batch/fit sets
    Phi_batch = Phi_all_d(id,:);          % (Nbatch x P)
    Phi_fit   = Phi_batch(idx_fit,:);     % (|fit| x P)
    dz_fit    = dz(idx_fit,:);
    x_fit     = xop(idx_fit,:);
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

    % Sample weights (taper near the guard)
    band_w = 0.30;
    dist_raw = abs(abs(raw_fit(:,1)) - 1);
    w_fit = min(1, dist_raw / band_w);

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

    % Resolve permutation on fit set (if gt labels available)
    if ~isempty(true_mode)
        m1 = nnz(mode_op(idx_fit) ~= true_mode(idx_fit));
        m2 = nnz((3 - mode_op(idx_fit)) ~= true_mode(idx_fit));
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
        [sol_poly, prog_poly, ~, vars_poly] = searchPoly(x_fit, dz_fit, lambda_used(idx_fit,:), prog_poly, Phi_fit);
    else
        [sol_poly, prog_poly, ~, vars_poly] = searchPoly(xop,  dz,    lambda_used,                prog_poly, Phi_batch);
    end

    a1 = double(sol_poly.eval(vars_poly.a1));
    b1 = double(sol_poly.eval(vars_poly.b1));
    a2 = double(sol_poly.eval(vars_poly.a2));
    b2 = double(sol_poly.eval(vars_poly.b2));

    cost_poly_log(iter) = double(sol_poly.eval(sum(vars_poly.delta) * 1e3));

    % External diagnostics (full batch, stable)
    [L1s, L1h] = eval_l1_losses(Phi_batch, dz, lambda_soft, a1,b1,a2,b2);
    fprintf('Iter %2d | L1 soft=%.3e | L1 hard=%.3e | mismatch=%d | poly=%.3e\n', ...
        iter, L1s, L1h, mode_error(iter), cost_poly_log(iter));

    % Validation mismatch (RAW guard on fixed set)
    Phi_val = buildMonomialMatrix(Xn(id_val,:), d);
    H1v = [Phi_val*a1, Phi_val*b1];
    H2v = [Phi_val*a2, Phi_val*b2];
    prog_val = spotsosprog;
    [sol_val, ~, ~, vars_val] = searchMode(dz_valn, {H1v, H2v}, prog_val);
    lam_val = double(sol_val.eval(vars_val.lam));
    [~, mode_val] = max(lam_val, [], 2);
    val_error(iter) = nnz(mode_val ~= gt_val); %#ok<SAGROW>
end

%% ----------------- Learn switching surface -----------------
eta=10; epsilon=1e-2; beta=1e-2;
sigma = 2*(mode_op==1) - 1;
[a_opt, z_val, diagnostics] = searchSwitchingSurface_softmargin(xop, sigma, kappa, eta, epsilon, beta); %#ok<ASGLU>

%% ----------------- Quick plots -----------------
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
[x1r,x2r] = meshgrid(linspace(-3.5,3.5,300), linspace(-3.5,3.5,300));
Xgrid_raw = [x1r(:), x2r(:)];
Xgrid_n   = (Xgrid_raw - mx)./sx;
Phi_grid  = buildMonomialMatrix(Xgrid_n, kappa);
Z         = reshape(Phi_grid*a_opt, size(x1r));
figure(2); clf; hold on; axis equal; grid on; box on;
h1=scatter(xop_raw(idx1,1),xop_raw(idx1,2),10,[.9 .7 0],'filled','DisplayName','Mode 1');
h2=scatter(xop_raw(idx2,1),xop_raw(idx2,2),10,[0 0 .8],'filled','DisplayName','Mode 2');
[~,h3]=contour(x1r,x2r,Z,[0 0],'k','LineWidth',2); h3.DisplayName='$f(x)=0$';
xline(1,'r--'); xline(-1,'r--');
legend([h1 h2 h3],'Location','best'); xlabel('$x_1$'); ylabel('$x_2$');
title('Identified vs. true surface (raw)');

%% ----------------- Vector field sanity stats (optional) -----------------
mu_true = mu;
Xb = X(id,:);                         % raw batch states
Vgt = sps_true_rhs_vec(Xb, mu_true);
V_or = sps_learned_rhs_oracle(     Xb, a1,b1,a2,b2, d, mx,sx, md,sd);
V_le = sps_learned_rhs_learnedguard(Xb, a1,b1,a2,b2, d, a_opt, kappa, mx,sx, md,sd);
MAE_or = mean(abs(V_or - Vgt),1); MAE_le = mean(abs(V_le - Vgt),1);
ANG_or = mean(ang_err_deg(V_or, Vgt)); ANG_le = mean(ang_err_deg(V_le, Vgt));
fprintf('\nBatch stats | MAE(true)=%.2e,%.2e / MAE(learned)=%.2e,%.2e | ANG=%.2f/%.2f deg\n', ...
    MAE_or(1),MAE_or(2), MAE_le(1),MAE_le(2), ANG_or, ANG_le);

%% ----------------- EXPORT: normalized + RAW -----------------
% Sanity: exps sizes must match learned coeff counts
assert(size(Phi_all_d,2)==numel(a1) && numel(a1)==numel(b1) && numel(a1)==numel(a2) && numel(a1)==numel(b2), ...
    'Inconsistent coefficient lengths vs Phi columns.');
if ~isempty(exps_d)
    assert(size(exps_d,1)==numel(a1), 'exps_d row count must match coeff length.');
end
if ~isempty(exps_k)
    assert(size(exps_k,1)==numel(a_opt), 'exps_k row count must match surface coeff length.');
end

% Evaluate surface and make hard labels from sign
Xn_all    = (X - mx) ./ sx;
Phi_all_k = buildMonomialMatrix(Xn_all, kappa);
fX        = Phi_all_k * a_opt;
mode_f    = ones(size(fX)); mode_f(fX < 0) = 2;
sigma_f   = 2*(mode_f==1) - 1;

% Per-mode dynamics on all data (normalized → RAW)
Phi_all_d = buildMonomialMatrix(Xn_all, d);
VH1n = [Phi_all_d*a1, Phi_all_d*b1];
VH2n = [Phi_all_d*a2, Phi_all_d*b2];
VH1  = VH1n .* sd + md;   % RAW
VH2  = VH2n .* sd + md;
C_all = [sum((Xdot - VH1).^2,2), sum((Xdot - VH2).^2,2)];
[~, mode_resid] = min(C_all, [], 2);

% Model struct (normalized-input representation)
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

% Symbolic (optional)
try
    syms x1 x2 real
    f_sym = sym(0);
    if ~isempty(exps_k)
        for j=1:size(exps_k,1), f_sym = f_sym + a_opt(j)*x1^exps_k(j,1)*x2^exps_k(j,2); end
        model.surface.f_symbolic = char(simplify(f_sym));
    end
    f1m1=sym(0); f2m1=sym(0); f1m2=sym(0); f2m2=sym(0);
    if ~isempty(exps_d)
        for j=1:size(exps_d,1)
            mon = x1^exps_d(j,1)*x2^exps_d(j,2);
            f1m1 = f1m1 + a1(j)*mon;  f2m1 = f2m1 + b1(j)*mon;
            f1m2 = f1m2 + a2(j)*mon;  f2m2 = f2m2 + b2(j)*mon;
        end
    end
    model.modes(1).dx1_symbolic = char(simplify(f1m1));
    model.modes(1).dx2_symbolic = char(simplify(f2m1));
    model.modes(2).dx1_symbolic = char(simplify(f1m2));
    model.modes(2).dx2_symbolic = char(simplify(f2m2));
catch
    model.surface.f_symbolic = 'Symbolic Math Toolbox not available';
end

% Per-sample diag
model.full.X = X; model.full.Xdot = Xdot;
model.full.fX = fX; model.full.mode_fsign = mode_f;
model.full.mode_resid = mode_resid;
model.metadata.created = datestr(now,'yyyy-mm-ddTHH:MM:SS');
model.metadata.description = 'Bilevel SPS ID with polynomial surface (hard law).';

% Save MAT/JSON
save('sps_identified_model.mat','model');
fid=fopen('sps_identified_model.json','w'); fwrite(fid,jsonencode(model)); fclose(fid);

% CSV: per-mode coeffs (normalized basis, correct exps order)
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

% CSV: switching surface (normalized inputs)
if ~isempty(exps_k)
    writetable(table(exps_k(:,1), exps_k(:,2), a_opt(:), ...
        'VariableNames',{'e1','e2','a'}), 'sps_switch_surface.csv');
end

% CSV: full dataset + f(x) + hard labels
writetable(table(X(:,1),X(:,2),Xdot(:,1),Xdot(:,2),fX,mode_f,sigma_f, ...
    'VariableNames',{'x1','x2','x1dot','x2dot','fX','mode_fsign','sigma'}), ...
    'sps_full_data_with_modes.csv');

% Expand surface to RAW and dynamics to RAW (proper binomial + de-norm)
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

%% ======================= Helpers =======================
function [L1_soft, L1_hard] = eval_l1_losses(Phi, dz, lambda_soft, a1,b1,a2,b2)
    H1 = [Phi*a1, Phi*b1];
    H2 = [Phi*a2, Phi*b2];
    Hs = lambda_soft(:,1).*H1 + lambda_soft(:,2).*H2;
    L1_soft = sum(sum(abs(dz - Hs), 2));
    [~, mh] = max(lambda_soft, [], 2);
    Hh = H1; Hh(mh==2,:) = H2(mh==2,:);
    L1_hard = sum(sum(abs(dz - Hh), 2));
end

function [sol, prog, cost_vec, vars] = searchMode_weighted(dz, dz_tilde, prog_base, w)
    % Weighted L1 simplex LP
    N = size(dz,1); M = numel(dz_tilde); n = size(dz,2);
    [prog, lam] = prog_base.newPos(N*M); 
    lam = reshape(lam,[N,M]);

    cost_vec = [];
    for k=1:N
        dzk = 0;
        for q=1:M, dzk = dzk + dz_tilde{q}(k,:)' * lam(k,q); end
        cost_vec = [cost_vec; dz(k,:)' - dzk]; %#ok<AGROW>
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
end

function V = sps_true_rhs_vec(Xraw, mu_)
    x1 = Xraw(:,1); x2 = Xraw(:,2);
    xterm = x1;
    mask = abs(x1) <= 1; xterm(mask) = x1(mask).^3;
    V = [ x2, mu_*(1 - x1.^2).*x2 - xterm ];
end

function Vraw = sps_learned_rhs_oracle(Xraw, a1,b1,a2,b2,deg, mx,sx, md,sd)
    Xn = (Xraw - mx)./sx;
    Phi = buildMonomialMatrix(Xn,deg);
    V1n = [Phi*a1, Phi*b1]; V2n = [Phi*a2, Phi*b2];
    Vn  = V2n; Vn(abs(Xraw(:,1)) <= 1,:) = V1n(abs(Xraw(:,1)) <= 1,:);
    Vraw = Vn.*sd + md;
end

function Vraw = sps_learned_rhs_learnedguard(Xraw, a1,b1,a2,b2,deg, a_opt, kappa, mx,sx, md,sd)
    Xn = (Xraw - mx)./sx;
    Phi_k = buildMonomialMatrix(Xn,kappa);
    fn = Phi_k * a_opt;
    Phi = buildMonomialMatrix(Xn,deg);
    V1n = [Phi*a1, Phi*b1]; V2n = [Phi*a2, Phi*b2];
    Vn  = V2n; Vn(fn >= 0,:) = V1n(fn >= 0,:);
    Vraw = Vn.*sd + md;
end

function ang = ang_err_deg(Vhat, Vtrue)
    epsn = 1e-12;
    c = sum(Vhat.*Vtrue,2) ./ max(vecnorm(Vhat,2,2).*vecnorm(Vtrue,2,2), epsn);
    c = max(-1,min(1,c));
    ang = acosd(c);
end

function T = local_expand_surface_to_raw(a_opt, exps_k, mx, sx)
    % Expand f((x-mx)./sx) into RAW polynomial coefficients
    if isempty(exps_k)
        T = table([],[],[], 'VariableNames',{'e1','e2','a_raw'});
        return;
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
        e1r(end+1,1) = pq(1); %#ok<AGROW>
        e2r(end+1,1) = pq(2); %#ok<AGROW>
        ar(end+1,1)  = Araw(keysCell{kk}); %#ok<AGROW>
    end
    T = sortrows(table(e1r,e2r,ar,'VariableNames',{'e1','e2','a_raw'}), {'e1','e2'});
end


function T = local_expand_dyn_to_raw(coeffs, exps_d, mx, sx, out_gain, out_bias)
    % v_raw(x) = out_gain * sum c_ij ((x1-mx1)/sx1)^i ((x2-mx2)/sx2)^j + out_bias
    if isempty(exps_d)
        error('exps_d required to label RAW coefficients.');
    end
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
    % Add output bias to constant term
    key00 = '0_0';
    if isKey(Araw, key00), prev = Araw(key00); else, prev = 0.0; end
    Araw(key00) = prev + out_bias;

    keysCell = Araw.keys();
    e1r=[]; e2r=[]; cr=[];
    for kk = 1:numel(keysCell)
        pq = sscanf(keysCell{kk}, '%d_%d');
        e1r(end+1,1) = pq(1); %#ok<AGROW>
        e2r(end+1,1) = pq(2); %#ok<AGROW>
        cr(end+1,1)  = Araw(keysCell{kk}); %#ok<AGROW>
    end
    T = sortrows(table(e1r,e2r,cr,'VariableNames',{'e1','e2','coeff_raw'}), {'e1','e2'});
end

