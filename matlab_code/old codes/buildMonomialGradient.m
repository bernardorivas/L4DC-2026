function grad_phi = buildMonomialGradient(x, d)
% Computes the gradient ∇φ(x) of all monomials up to degree ≤ d
% Input:  x - n×1 vector
% Output: grad_phi - P × n matrix (∂φ_j/∂x_k)

    x = x(:);  % Ensure column
    n = length(x);
    exp_list = generateExponentList(n, d);  % P × n
    exp_list = fliplr(exp_list);  % Make consistent with buildMonomialMatrix
    P = size(exp_list, 1);

    grad_phi = zeros(P, n);
    for i = 1:P
        alpha = exp_list(i, :);
        for j = 1:n
            if alpha(j) == 0
                grad_phi(i, j) = 0;
            else
                new_alpha = alpha;
                new_alpha(j) = new_alpha(j) - 1;
                grad_phi(i, j) = alpha(j) * prod(x'.^new_alpha);
            end
        end
    end
end

function exponents = generateExponentList(n, d)
    % Generates all exponent vectors of total degree ≤ d in n variables
    exponents = [];
    for total_deg = 0:d
        exponents = [exponents; generateHomogeneousExponents(n, total_deg)];
    end
end

function exponents = generateHomogeneousExponents(n, total_deg)
    if n == 1
        exponents = total_deg;
    else
        exponents = [];
        for k = 0:total_deg
            tails = generateHomogeneousExponents(n - 1, total_deg - k);
            exponents = [exponents; [k * ones(size(tails,1),1), tails]];
        end
    end
end