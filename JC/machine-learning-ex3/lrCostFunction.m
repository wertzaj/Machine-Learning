function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================

h = sigmoid(X*theta); %sigmoid func does this elementwise

J = 1/m*( -y'*log(h) - (1-y)'*log(1-h) ) + lambda/(2*m)*sum(theta(2:end).^2);
% We need to skip theta0 ie theta(1) since the bias node is not regularized
% The resulted value to the right of the '+' is a scalar value

%=============== METHOD 1
% I think I would need to create a temp_sigmoid funct if I reindex here
% grad = 1/m * X'*(h-y);
% grad(2:end) = grad + 1/m * X(:,2:end)'*(h-y) + lambda/m*temp;

%=============== METHOD 2
grad = 1/m * X'*(h-y);
temp = theta;
temp(1) = 0;
grad = grad + lambda/m*temp;

% ====================== EXPLANATION HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
% =============================================================

grad = grad(:);

end
