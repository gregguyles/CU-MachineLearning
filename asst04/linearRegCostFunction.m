% Greg Guyles
% Machine Learning
% Asst 3
% 2-13-14

function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% find hypothesis
hyp = X * theta;

% cost function
J = sum( ( hyp - y) .^ 2 ) / ( 2 * m );

% regularize
thetaLen = length(theta);
if (thetaLen >=2)
    J = J + (lambda / ( 2 * m) ) * ( sum ( theta(2:thetaLen, 1) .^ 2 ) );
end

% gradient function
grad = (X' * bsxfun(@minus, hyp, y)) ./ m;
if (thetaLen >= 2)
    grad(2:thetaLen, 1) = grad(2:thetaLen, 1) + ...
        ( lambda / m ) .* theta(2:thetaLen, 1);
end
% =========================================================================

grad = grad(:);

end
