function [a_opt, xi_val, diagnostics] = searchSwitchingSurface_softmargin(X, sigma, kappa, eta, epsilon, beta)
% Inputs:
%   X       : N x 2 data points
%   sigma   : N x 1 vector of labels in {+1, -1}
%   kappa   : total polynomial degree
%   eta     : box constraint bound on coefficients a
%   epsilon : soft margin threshold
%   beta    : regularization strength on ||a||_1

    % Build monomial matrix
    Phi = buildMonomialMatrix(X, kappa);  % N x P
    [N, P] = size(Phi);

    % Variables
    a = sdpvar(P,1);         % polynomial coefficients
    xi = sdpvar(N,1);        % soft margin slack variables
    t  = sdpvar(P,1);        % abs(a_j) for L1 regularization

    % Constraints
    constraints = [];
    for i = 1:N
        constraints = [constraints;
            sigma(i) * (Phi(i,:) * a) >= epsilon - xi(i)
        ];
    end

    constraints = [constraints;
        xi >= 0;
        -eta <= a <= eta;
        t >= a;
        t >= -a;
        % sum(a) == 1; % Optional normalization if needed
    ];

    % Objective: hinge loss + L1 penalty
    objective = sum(xi) + beta * sum(t);

    % Solver settings
    options = sdpsettings('verbose', 1, 'solver', 'intlinprog');  % or 'gurobi' if installed

    % Solve
    diagnostics = optimize(constraints, objective, options);

    % Return solution
    if diagnostics.problem == 0
        a_opt = value(a);
        xi_val = value(xi);
    else
        warning("Solver failed: %s", diagnostics.info);
        a_opt = NaN(P,1);
        xi_val = NaN(N,1);
    end
end