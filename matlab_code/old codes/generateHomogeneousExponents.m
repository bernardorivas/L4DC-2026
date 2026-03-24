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