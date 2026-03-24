function Xdot_pred = predict_vel_from_soft_modes_poly(X, lambda_soft, coeffs, kappa)
% X: [N x n], lambda_soft: [N x Q], rows sum to 1
% coeffs{q}{ell}: P×1
% kappa: polynomial degree

[N, n] = size(X);
Q = size(lambda_soft, 2);
Phi = buildMonomialMatrix(X, kappa);   % [N x P]

Xdot_pred = zeros(N, n);
for q = 1:Q
    Vq = zeros(N, n);
    for ell = 1:n
        Vq(:, ell) = Phi * coeffs{q}{ell};
    end
    % elementwise weight by λ_q and accumulate
    Xdot_pred = Xdot_pred + (lambda_soft(:, q) .* ones(1, n)) .* Vq;
end
end
