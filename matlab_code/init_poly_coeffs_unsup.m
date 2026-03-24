function [coeffs, lambda_soft, flat] = init_poly_coeffs_unsup(X, Xdot, kappa, Q, varargin)
% Unsupervised init for switching polynomial vector fields (no labels).
% Fits Q polynomial vector fields by alternating:
%   - Weighted ridge LS per mode
%   - Soft responsibilities from 1-norm residuals via softmin(τ)
%
% Inputs:
%   X      : (N x n)
%   Xdot   : (N x n)
%   kappa  : polynomial degree for buildMonomialMatrix
%   Q      : # modes (use Q=2 here)
% Options:
%   'Rounds' (default 5)     : EM rounds
%   'Ridge'  (default 1e-6)  : ridge L2 regularization
%   'Tau'    (default 1e-2)  : softmin temperature
%   'Seed'   (default 'global+noise') : {'global+noise','random'}
%
% Outputs:
%   coeffs{q}{ell} : P x 1 per mode q and state-dim ell
%   lambda_soft    : (N x Q), rows sum to 1
%   flat           : struct fields a1,b1,a2,b2 when Q=2,n=2 (for searchPoly)

p = inputParser;
p.addParameter('Rounds', 5, @isscalar);
p.addParameter('Ridge', 1e-6, @isscalar);
p.addParameter('Tau', 1e-2, @isscalar);
p.addParameter('Seed', 'global+noise', @(s)ischar(s) || isstring(s));
p.parse(varargin{:});
Rounds = p.Results.Rounds;
ridge  = p.Results.Ridge;
tau    = p.Results.Tau;
seed   = string(p.Results.Seed);

[N, n] = size(X);
Phi    = buildMonomialMatrix(X, kappa);   % (N x P)
P      = size(Phi,2);

% -------- seed coeffs and λ (no labels) --------
coeffs = cell(Q,1);
switch seed
case "global+noise"
    % Fit one global poly to all data, duplicate to Q modes with tiny noise
    coeffs_global = cell(n,1);
    H = (Phi.'*Phi) + ridge*speye(P);
    for ell = 1:n
        rhs = Phi.'*Xdot(:,ell);
        coeffs_global{ell} = H \ rhs;
    end
    for q = 1:Q
        coeffs{q} = cell(n,1);
        for ell = 1:n
            coeffs{q}{ell} = coeffs_global{ell} + 1e-3*randn(P,1);
        end
    end
    lambda_soft = (1/Q) * ones(N,Q);           % uniform responsibilities
case "random"
    for q = 1:Q
        coeffs{q} = cell(n,1);
        for ell = 1:n
            coeffs{q}{ell} = randn(P,1);
        end
    end
    z = rand(N,Q); lambda_soft = z./sum(z,2);
otherwise
    error('Unknown Seed option.');
end

% -------- EM-style refinement --------
for r = 1:Rounds
    % (E) responsibilities from residuals
    R = zeros(N,Q);    % 1-norm residuals per mode
    for q = 1:Q
        Vq = zeros(N,n);
        for ell = 1:n
            Vq(:,ell) = Phi * coeffs{q}{ell};
        end
        R(:,q) = sum(abs(Xdot - Vq), 2);   % L1 residual across components
    end
    % softmin: λ_{i,q} = exp(-R_{i,q}/τ)/sum_q exp(...)
    Z = exp(-R/max(tau,1e-12));
    lambda_soft = Z ./ sum(Z,2);

    % (M) weighted ridge LS per mode
    for q = 1:Q
        w = lambda_soft(:,q);                       % N x 1 weights
        W = spdiags(w, 0, N, N);                    % N x N
        AtW = Phi.' * W;                            % P x N
        H   = AtW * Phi + ridge * speye(P);         % P x P
        for ell = 1:n
            rhs               = AtW * Xdot(:,ell);  % P x 1
            coeffs{q}{ell}    = H \ rhs;
        end
    end
end

% -------- flatten for Q=2, n=2 convenience (searchPoly) --------
flat = struct();
if Q==2 && n==2
    flat.a1 = coeffs{1}{1}; flat.b1 = coeffs{1}{2};
    flat.a2 = coeffs{2}{1}; flat.b2 = coeffs{2}{2};
end
end
