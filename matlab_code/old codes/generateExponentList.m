function exponents = generateExponentList(n, d)
    % Generates all exponent vectors of total degree ≤ d in n variables
    exponents = [];
    for total_deg = 0:d
        exponents = [exponents; generateHomogeneousExponents(n, total_deg)];
    end
end