function Xdot_pred = predict_vel_from_modes_poly(X, mode_pred, coeffs, kappa)
% X: [N x n], mode_pred in {1,2,...,Q}
% coeffs{q}{ell} are P x 1 polynomial coefficients for state-dim ell
% kappa = poly degree used to build monomial basis

[N, n] = size(X);
Q = numel(coeffs);

Phi = buildMonomialMatrix(X, kappa);  % [N x P]
Fq  = cell(Q,1);
for q = 1:Q
    Vq = zeros(N,n);
    for ell = 1:n
        Vq(:,ell) = Phi * coeffs{q}{ell};
    end
    Fq{q} = Vq;
end

Xdot_pred = zeros(N,n);
for q = 1:Q
    mask = (mode_pred == q);
    if any(mask)
        Xdot_pred(mask,:) = Fq{q}(mask,:);
    end
end
end
