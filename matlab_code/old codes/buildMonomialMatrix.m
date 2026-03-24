function Phi = buildMonomialMatrix(X, kappa)
    % X is N x n, builds monomial basis matrix Φ of total degree ≤ kappa
    [N, n] = size(X);
    exp_list = generateExponentList(n, kappa);  % P x n
    exp_list = fliplr(exp_list);  % <-- Fix: reverse so that columns match x1, x2, ..., xn
    P = size(exp_list, 1);

    % Initialize Phi
    Phi = ones(N, P);
    for j = 1:P
        for k = 1:n
            Phi(:, j) = Phi(:, j) .* X(:, k).^exp_list(j, k);
        end
    end
end
