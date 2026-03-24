%% bilevel_sls4_id.m  (Q=4, factored+margin mode LP + affine update)
% xdot ≈ Σ_q λ_{iq} (A_q x_i + b_q),  λ_i factored via (u,v,w) McCormick

clear; clc; close all;

% -------- Load data --------
% (Uncomment exactly one)
%load sls_bistability_data.mat      % expects: data.y (N×2), data.dy (N×2), data.mode (opt)
load toggle_switch_GT_data.mat      % same structure

X        = double(data.y);           % N×n
Xdot     = double(data.dy);          % N×n
trueMode = [];
if isfield(data,'mode') && ~isempty(data.mode), trueMode = data.mode(:); end

[N, n] = size(X);
Q = 4;  assert(n==2,'This script assumes n=2.');

% -------- Hyperparameters --------
nIter   = 5;
batchN  = min(N, 600);       % batching is stratified and safe if N<500
rng(0);

% -------- Logs --------
cost_matrix_log = zeros(nIter,1);
mode_error      = zeros(nIter,1);

% ================================================================
% Unsupervised warm-start of {A_q, b_q} (NO thresholds, NO labels)
% ================================================================
rep = 5;                                      % a few restarts helps
[idx0, C] = kmeans(X, Q, 'Replicates', rep, 'MaxIter', 500);

% Reproducible cluster→mode guess by x1 position (simple symmetry break)
[~, ix] = sort(C(:,1));
perm_guess = 1:Q;
perm_guess(ix(1:2)) = [1 3];   % left side (low x) → assign 1,3
perm_guess(ix(3:4)) = [2 4];   % right side (high x) → assign 2,4
labels0 = perm_guess(idx0);    % relabeled k-means clusters (optional)
labels0 = labels0(:);          % ensure column

A = zeros(n,n,Q);
b = zeros(n,Q);
for q = 1:Q
    Iq = find(labels0==q);
    if numel(Iq) >= max(4, n+2)   % tiny guard
        Xq    = X(Iq,:);
        Xdotq = Xdot(Iq,:);
        Phi   = [Xq, ones(numel(Iq),1)];
        theta = Phi \ Xdotq;                 % (n+1)×n
        A(:,:,q) = theta(1:n,:).';           % n×n
        b(:,q)   = theta(n+1,:).';           % n×1
    else
        A(:,:,q) = eye(n);
        b(:,q)   = zeros(n,1);
    end
end

% Running label estimate for stratified batching (updated each iter)
labels_prev = labels0;  % column

%% ==== Main alternating loop ====
for iter = 1:nIter

    % ------------------------------------------------------------
    % Stratified batching by current label estimate (NO thresholds)
    % ------------------------------------------------------------
    Bq  = max(1, floor(batchN/Q));
    idx = zeros(0,1);  % column

    for qk = 1:Q
        Iq = find(labels_prev==qk);      % column
        take = min(Bq, numel(Iq));
        if take > 0
            sel = Iq(randperm(numel(Iq), take));
            idx = [idx; sel(:)];         %#ok<AGROW>
        end
    end

    % top-up if short
    if numel(idx) < batchN
        rest  = setdiff((1:N).', idx, 'stable');   % column
        extra = min(batchN - numel(idx), numel(rest));
        if extra > 0
            add = rest(randperm(numel(rest), extra));
            idx = [idx; add(:)];                   %#ok<AGROW>
        end
    end

    % shuffle (keep column)
    idx = idx(randperm(numel(idx)));

    % ---- Define batch variables ----
    xop  = X(idx, :);                 % (B×n)
    dz   = Xdot(idx, :);              % (B×n)
    tmod = [];
    if ~isempty(trueMode), tmod = trueMode(idx); end
    B    = size(xop,1);

    % ===== Build per-mode predictions =====
    dy_tilde = cell(1, Q);
    for q = 1:Q
        dy_tilde{q} = xop * A(:,:,q).' + b(:,q).';   % (B×n)
    end

    % ===== Winner hints from current residuals =====
    Rhint = zeros(B, Q);
    for q = 1:Q
        Rhint(:,q) = sum(abs(dz - dy_tilde{q}), 2);
    end
    [~, q_hint] = min(Rhint, [], 2);

    % ===== MODE ASSIGNMENT: factored + margin sharpening (LP) =====
    rho0 = 3e-3; rhoT = 8e-3;                               % anneal a bit
    rho  = rho0 * (rhoT/rho0)^((iter-1)/max(1,nIter-1));

    prog_mode = spotsosprog;
    [sol_mode, prog_mode, ~, vars_mode] = searchMode_factored_peak(dz, dy_tilde, prog_mode, q_hint, rho);

    lam_soft = double(sol_mode.eval(vars_mode.lam));        % (B×4)
    [~, mode_op] = max(lam_soft, [], 2);                    % hard labels ∈ {1..4}
    lambda_hard  = sparse(1:B, mode_op, 1, B, Q);           % rows sum to 1

    % Update running labels for future stratified batches (data indices `idx`)
    labels_prev(idx) = mode_op;

    % ----- optional: relabel to match provided GT (for metric only)
    if ~isempty(tmod)
        P = perms(1:Q);
        mis = inf(size(P,1),1);
        for k = 1:size(P,1)
            mis(k) = nnz(P(k, mode_op).' ~= tmod);
        end
        [~, kbest] = min(mis);
        map = P(kbest,:);
        mode_error(iter) = nnz(map(mode_op).' ~= tmod);
    end

    % ===== UPDATE (A_q, b_q) with fixed λ (ℓ1 LP) =====
    prog_matrix = spotsosprog;
    [solM, prog_matrix, ~, varsM] = searchMatrixAffine(xop, dz, lambda_hard, prog_matrix);

    % Extract A, b
    for q = 1:Q
        A(:,:,q) = reshape(double(solM.eval(varsM.a{q})), n, n);
        b(:,q)   = double(solM.eval(varsM.b{q}));
    end

    % Log ℓ1 fit on batch (unscaled sum of slacks)
    cost_matrix_log(iter) = double(solM.eval(sum(varsM.delta)));

    % Diagnostics
    counts = full(sum(lambda_hard,1));
    fprintf('Iter %2d | L1 batch: %.3e | counts: [%d %d %d %d] | rho=%.2e | mismatch: %d\n', ...
            iter, cost_matrix_log(iter), counts(1),counts(2),counts(3),counts(4), rho, mode_error(iter));
end

%% ===== Simple diagnostics =====
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

figure(1); clf; hold on; box on; grid on;
plot(abs(cost_matrix_log), '-o', 'LineWidth', 1.5); set(gca,'YScale','log');
xlabel('Iteration'); ylabel('$\sum |r|$ on batch'); title('Affine parameter fit (L1)');

if any(mode_error)
    figure(2); clf; hold on; box on; grid on;
    plot(mode_error, '-s', 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('Mismatch count'); title('Mode assignment error');
end

% Print final parameters
for q = 1:Q
    fprintf('\n=== Mode %d ===\nA_%d = \n', q, q);
    disp(A(:,:,q));
    fprintf('b_%d^T = [%.6g  %.6g]\n', q, b(1,q), b(2,q));
end

%% ===== Residual-based hard modes on ALL data (final sanity) =====
R = zeros(N,Q);
for q = 1:Q
    R(:,q) = sum(abs(Xdot - (X*A(:,:,q).'+b(:,q).')), 2);
end
[~, mode_resid] = min(R, [], 2);

figure(3); clf; hold on; grid on; axis equal;
cols = lines(Q);
for q = 1:Q
    idxq = (mode_resid == q);
    scatter(X(idxq,1), X(idxq,2), 8, cols(q,:), 'filled');
end
xlabel('$x_1$'); ylabel('$x_2$'); title('Residual-based hard modes (final params)');
legend(arrayfun(@(q)sprintf('Mode %d',q),1:Q,'uni',0),'Location','best');

%% ===== Switching surface recovery with L=2 polynomials (Q=4) =====
% Use mode-book consistent with construct.m: k=1..4 ↔ (-,-),(+,-),(-,+),(+,+)

% ---------- 1) Choose labels for training ----------
labels_final = mode_resid(:);      % N×1 in {1,2,3,4}
Nall = size(X,1);

% ---------- 2) Mode-book (rows = sign pairs for modes 1..4) ----------
Sbook = [ -1, -1;   % k=1 : (i,j)=(0,0)
          +1, -1;   % k=2 : (1,0)
          -1, +1;   % k=3 : (0,1)
          +1, +1];  % k=4 : (1,1)

% ---------- 3) Convert modes → binary targets sigma_{i,ell} ----------
sigma = zeros(Nall, 2);
for j = 1:4
    idxj = (labels_final == j);
    if any(idxj)
        sigma(idxj, :) = repmat(Sbook(j, :), nnz(idxj), 1);
    end
end

% ---------- 4) Solve two independent soft-margin problems ----------
kappa   = 1;       % polynomial degree (tune)
eta     = 10;      % box bound in your routine
epsilon = 1e-2;    % margin
beta    = 1e-2;    % sparsity weight

[a1_opt, z1_val] = searchSwitchingSurface_softmargin(X, sigma(:,1), kappa, eta, epsilon, beta);
[a2_opt, z2_val] = searchSwitchingSurface_softmargin(X, sigma(:,2), kappa, eta, epsilon, beta);

% ---------- 5) Predict regions by sign pair ----------
Phi_all = buildMonomialMatrix(X, kappa);
f1 = Phi_all * a1_opt;
f2 = Phi_all * a2_opt;

% Robust sign with small tolerance
tau = 1e-9;
s1 = ones(Nall,1); s1(f1 < -tau) = -1;
s2 = ones(Nall,1); s2(f2 < -tau) = -1;
S  = [s1, s2];

% Map sign pairs back to mode indices via the mode-book
key = containers.Map({'- -','+ -','- +','+ +'}, num2cell(1:4));
mode_surface = zeros(Nall,1);
for i = 1:Nall
    kstr = sprintf('%+d %+d', S(i,1), S(i,2));
    kstr = strrep(kstr,'+1','+'); kstr = strrep(kstr,'-1','-');
    mode_surface(i) = key(kstr);
end

% ---------- 6) Sanity metrics ----------
mismatch_vs_resid = nnz(mode_surface ~= labels_final);
fprintf('Surfaces (L=2) vs residual modes mismatch: %d / %d\n', mismatch_vs_resid, Nall);

% ---------- 7) Visualization ----------
x1r = linspace(min(X(:,1)), max(X(:,1)), 300);
x2r = linspace(min(X(:,2)), max(X(:,2)), 300);
[XX,YY] = meshgrid(x1r, x2r);
Phi_grid = buildMonomialMatrix([XX(:),YY(:)], kappa);
Z1 = reshape(Phi_grid * a1_opt, size(XX));
Z2 = reshape(Phi_grid * a2_opt, size(XX));

cols = lines(4);

% (a) Each surface's zero set
figure(21); clf; tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile; hold on; axis equal; box on; grid on;
contour(XX,YY,Z1,[0 0],'k','LineWidth',1.5);
scatter(X(:,1), X(:,2), 6, 'filled'); title('$f_1(x)=0$','Interpreter','latex');
xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex');

nexttile; hold on; axis equal; box on; grid on;
contour(XX,YY,Z2,[0 0],'k','LineWidth',1.5);
scatter(X(:,1), X(:,2), 6, 'filled'); title('$f_2(x)=0$','Interpreter','latex');
xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex');

% (b) Combined: points colored by region (sign pair) and both contours
figure(22); clf; hold on; axis equal; box on; grid on;
for q = 1:4
    scatter(X(mode_surface==q,1), X(mode_surface==q,2), 6, cols(q,:), 'filled');
end
contour(XX,YY,Z1,[0 0],'k','LineWidth',1.25);
contour(XX,YY,Z2,[0 0],'k','LineWidth',1.25);
legend({'Mode 1','Mode 2','Mode 3','Mode 4'},'Location','best');
xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex');
%title('Two-surface partition: sign($f_1$), sign($f_2$)','Interpreter','latex');


%% ===== Mode-book visual checks (q = 1 + i + 2 j and flips) =====
% Uses learned surfaces f1, f2 from above and residual labels `labels_final`.

tau = 1e-9;                                  % sign tolerance
sgn = @(u) 2*(u>tau) - 1;                    % returns -1 or +1 with tie -> +1

% Helper: from signs to q under the canonical book q = 1 + i + 2j
assign_q = @(s1,s2) (1 + (s1>0) + 2*(s2>0)); % i=1 if f1>0, j=1 if f2>0

% Four schemes: [flip_f1, flip_f2]
schemes = [0 0; 1 0; 0 1; 1 1];
names   = {'canonical (q=1+i+2j)','flip f_1','flip f_2','flip f_1 & f_2'};

cols = lines(4);
x1r = linspace(min(X(:,1)), max(X(:,1)), 300);
x2r = linspace(min(X(:,2)), max(X(:,2)), 300);
[XX,YY]  = meshgrid(x1r, x2r);
Phi_grid = buildMonomialMatrix([XX(:),YY(:)], kappa);
Z1 = reshape(Phi_grid * a1_opt, size(XX));
Z2 = reshape(Phi_grid * a2_opt, size(XX));

figure(23); clf; tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

for k = 1:4
    flip1 = schemes(k,1);  flip2 = schemes(k,2);

    % flip surfaces if needed (for classification only; keep originals intact)
    s1 = sgn( (flip1==0)*f1 + (flip1==1)*(-f1) );
    s2 = sgn( (flip2==0)*f2 + (flip2==1)*(-f2) );

    % hard labels by q = 1 + i + 2j
    qk = assign_q(s1>0, s2>0);

    % mismatch vs residual-based labels
    mis = nnz(qk(:) ~= labels_final(:));

    nexttile; hold on; axis equal; box on; grid on;
    for q = 1:4
        scatter(X(qk==q,1), X(qk==q,2), 6, cols(q,:), 'filled');
    end

    % draw (possibly flipped) contours for readability
    Z1k = (flip1==0)*Z1 + (flip1==1)*(-Z1);
    Z2k = (flip2==0)*Z2 + (flip2==1)*(-Z2);
    contour(XX,YY,Z1k,[0 0],'k','LineWidth',1.1);
    contour(XX,YY,Z2k,[0 0],'k','LineWidth',1.1);

    title(sprintf('%s | mismatch: %d', names{k}, mis),'Interpreter','none');
    xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex');
    legend({'Mode 1','Mode 2','Mode 3','Mode 4'},'Location','bestoutside'); legend boxoff
end

%% ====================== EXPORTS ==========================
% Pack everything in a .mat (as before)
model2 = struct();
model2.n = 2; model2.Q = Q;
model2.A = A; model2.b = b;

model2.surface.L         = 2;
model2.surface.degree    = kappa;
model2.surface.coeffs    = {a1_opt(:), a2_opt(:)};   % f1, f2
model2.surface.mode_book = Sbook;                    % rows correspond to modes 1..4

% Exponent list matching buildMonomialMatrix ordering
exps = [];
try
    [~, exps] = buildMonomialMatrix(X(1:min(10,size(X,1)),:), kappa);
catch
    if exist('generateExponentList','file')==2
        exps = fliplr(generateExponentList(2, kappa));  % fallback, lex order
    else
        error('Could not obtain exponent list for surface export.');
    end
end
model2.surface.exponents = exps;

% Store full data and scores (optional)
model2.full.X     = X;
model2.full.Xdot  = Xdot;
model2.full.f1    = f1;
model2.full.f2    = f2;
model2.hard_modes.from_surfaces = mode_surface;
model2.hard_modes.from_residual = labels_final;

save('sls4_two_surface_model.mat','model2');

% --- (1) Surfaces CSV (long format): (surface, e1, e2, a) ---
S1 = table( repmat("f1", size(exps,1),1), exps(:,1), exps(:,2), a1_opt(:), ...
            'VariableNames', {'surface','e1','e2','a'} );
S2 = table( repmat("f2", size(exps,1),1), exps(:,1), exps(:,2), a2_opt(:), ...
            'VariableNames', {'surface','e1','e2','a'} );
Sall = [S1; S2];
writetable(Sall, 'sls4_surfaces_coeffs_long.csv');

% --- (2) Vector field coefficients CSV (long format) ---
rows_A = []; rows_b = [];
for q = 1:Q
    Aq = A(:,:,q); bq = b(:,q);
    [rA,cA] = ndgrid(1:size(Aq,1), 1:size(Aq,2));
    tblA = table( ...
        q*ones(numel(Aq),1), repmat("A",numel(Aq),1), rA(:), cA(:), Aq(:), ...
        'VariableNames', {'mode','type','row','col','value'} );
    tblb = table( ...
        q*ones(numel(bq),1), repmat("b",numel(bq),1), (1:numel(bq)).', zeros(numel(bq),1), bq(:), ...
        'VariableNames', {'mode','type','row','col','value'} );
    rows_A = [rows_A; tblA];
    rows_b = [rows_b; tblb];
end
vf_tbl = [rows_A; rows_b];
writetable(vf_tbl, 'sls4_vf_coeffs.csv');

disp('Export complete: sls4_two_surface_model.mat, sls4_surfaces_coeffs_long.csv, sls4_vf_coeffs.csv');
