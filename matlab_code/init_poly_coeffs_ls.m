function [coeffs, flat_coeffs] = init_poly_coeffs_ls(X, Xdot, kappa, varargin)
% Optional inputs:
%   'ModeInit', mode_init (Nx1 integers in {1..Q})
%   'LambdaSoft', lambda_soft (N x Q), rowsum=1
%   'Ridge', ridge (default 1e-6)
%
% Outputs:
%   coeffs{q}{ell} : P×1 per mode q and state dim ell
%   flat_coeffs : struct with fields a1,b1,a2,b2 (for Q=2,n=2 convenience)

p = inputParser;
p.addParameter('ModeInit', [], @(z)isvector(z) || isempty(z));
p.addParameter('LambdaSoft', [], @(z)ismatrix(z) || isempty(z));
p.addParameter('Ridge', 1e-6, @isscalar);
p.parse(varargin{:});
mode_init   = p.Results.ModeInit;
lambda_soft = p.Results.LambdaSoft;
ridge       = p.Results.Ridge;

[N, n] = size(X);
Phi = buildMonomialMatrix(X, kappa);  % [N x P]
P = size(Phi,2);

if isempty(lambda_soft)
    % Hard labels required
    if isempty(mode_init), error('Provide ModeInit or LambdaSoft.'); end
    Q = max(mode_init);
else
    Q = size(lambda_soft,2);
end

coeffs = cell(Q,1);
for q = 1:Q
    coeffs{q} = cell(n,1);
    % weights w: hard — indicator, soft — lambda(:,q)
    if isempty(lambda_soft)
        w = double(mode_init == q);
    else
        w = lambda_soft(:, q);
    end
    W = spdiags(w, 0, N, N);           % N×N diagonal
    A = Phi;                            % N×P
    for ell = 1:n
        b = Xdot(:, ell);               % N×1
        % Weighted ridge LS: (A'WA + ridge I) c = A'W b
        AtW  = A' * W;
        H    = AtW * A + ridge * speye(P);
        rhs  = AtW * b;
        coeffs{q}{ell} = H \ rhs;       % P×1
    end
end

% Convenience: flatten for Q=2, n=2 (a1,b1,a2,b2) to plug into searchPoly
flat_coeffs = struct();
if Q==2 && n==2
    flat_coeffs.a1 = coeffs{1}{1};
    flat_coeffs.b1 = coeffs{1}{2};
    flat_coeffs.a2 = coeffs{2}{1};
    flat_coeffs.b2 = coeffs{2}{2};
end
end
