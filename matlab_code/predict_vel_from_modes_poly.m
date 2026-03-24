function Xdot_pred = predict_vel_from_modes_poly(X, mode_pred, coeffs, kappa)
% X: [N x n], mode_pred ∈ {1,...,Q}
% coeffs{q}{ell}: P×1 polynomial coefficients for state dim ell
% kappa: polynomial degree used in buildMonomialMatrix

[N, n] = size(X);
Q = numel(coeffs);

Phi = buildMonomialMatrix(X, kappa);   % [N x P]
Xdot_pred = zeros(N, n);

for q = 1:Q
    mask = (mode_pred == q);
    if any(mask)
        Vq = zeros(sum(mask), n);
        for ell = 1:n
            Vq(:, ell) = Phi(mask,:) * coeffs{q}{ell};
        end
        Xdot_pred(mask, :) = Vq;
    end
end
end
