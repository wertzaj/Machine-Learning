function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
z1 = X*Theta1';
a2 = sigmoid(z1); % this will return a matrix 25x5000

a2_size = size(a2,1);   % add that bias term dawg
a2 = [ones(a2_size, 1) , a2];

z2 = a2*Theta2';
a3 = sigmoid(z2); % this will return a matrix 5000x10

% ===== Use max to find the integer ======================
% go to each row of p_temp and find the index with the highest value
% return/assign this index value to the p column vector
% there might be different ways to find max based on 'dim' value
% but we'll use 'dim=2' since it returns a col. vector
[perc, p] = max(a3, [], 2);
% =========================================================================

% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
% =========================================================================

end
