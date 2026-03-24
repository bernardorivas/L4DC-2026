function expMat = generateExponentMatrix(n, d)
% generateExponentMatrix: generate all exponent vectors of n variables up to total degree d
% Output: expMat is (P x n), where P = number of monomials

expMat = [];
for total = 0:d
    temp = partitions_n_sum(n, total);
    expMat = [expMat; temp];
end
end

function exponents = partitions_n_sum(n, total)
% Helper: All nonnegative integer solutions to x_1 + ... + x_n = total
if n == 1
    exponents = total;
else
    exponents = [];
    for k = 0:total
        tail = partitions_n_sum(n-1, total - k);
        exponents = [exponents; k * ones(size(tail,1),1), tail];
    end
end
end
