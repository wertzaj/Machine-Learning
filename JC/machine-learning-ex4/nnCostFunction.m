function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape( nn_params( 1:hidden_layer_size * (input_layer_size + 1) ) , ...
                 hidden_layer_size, (input_layer_size + 1));
% reshape( nn_params(1:10025), 25, 401);
% basically --> Theta1 is size 25 x 401
             
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
% reshape ( nn_params(10026:end, 10, 26);
% basically --> Theta2 is size 10 x 26
             
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
X_orig = X;
% Add a bias column to X matrix
rows_in_X = size(X,1); % 5000
temp = ones(rows_in_X,1);
X = [temp, X]; % so now, X --> 5000 x 401

a1 = X;                   % X --> 5000 x 401
a2 = sigmoid(a1*Theta1'); % a2 --> 5000 x 25
a2 = [ones(size(a2,1),1), a2];   % add that bias term dawg % a2 --> 5000 x 26
a3 = sigmoid(a2*Theta2'); % a3 --> 5000 x 10
% a3 is a matrix where each row is a single sample from the data set
% the columns of a3 hold the likehood (0:1) that that sample belongs to the
% corresponding class

% Now, we need to figure out how accurate a3 is. We'll do this by computing
% the Cost
% =========================================================================
% Method 1 - For Loop - Doesn't work all the way
% =========================================================================
j_temp = 0;
for i = 1:size(a3,1) % 1:5000
    % find the max likihood in the row and compare that index with the
    % true value. If it matches, then y = 1. If not, then y = 0.
    y_temp = zeros(size(a3,2), 1);
    [perc, p] = max(a3(i,:));
    % Return a Vector 'mask' of zeros with only one '1' that denotes the
    % correct index. ie, [0,1,0,0,] corresponds to a selection of '2' here -> [1,2,3,4]
    y_temp(p) = 1;
    
    for k = 1:size(a3,2) % 1:10
        j_temp = j_temp + -y_temp(k)*log(a3(i,k)) - (1-y_temp(k))*log(1-a3(i,k)); 
    end
end
J = 1/m*j_temp;

% =========================================================================
% Method 2 - Vectorized
% =========================================================================
% We're applying a mask to y so that we're wokring with 1's and 0's, not 10
% digits
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

% Compute the 'Cost' w/o the summation
J_temp = -y_matrix.*log(a3) - (1-y_matrix).*log(1-a3);

% Sum the matrix and dived my the num of samples
J = 1/m*sum(sum(J_temp));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

for t = 1:m
%     STEP 1 - OUTPUT LAYER
% don't include the bias term??
    %X = X_orig;
    a1 = X(t,:); % From the sample set, assign only one row at a time to the input nodes
    z2 = a1*Theta1';
    a2 = sigmoid(z2); % Do not compute Sigmoid with the bias term.
    a2 = [1 , a2];
    z3 = a2*Theta2';
    a3 = sigmoid(z3);
    
%     STEP 2 - Write out delta terms in a vector
    del3 = (a3 - y_matrix(t,:));
    
%     STEP 3 - HIDDEN LAYER
        % The key point to remember about BP is that the hidden layer bias unit
        % does not connect back to the input layer. So we do not include the first
        % column of Theta2 in the BP calculations that lead to Delta1 and
        % That1_grad
    del2 = del3*Theta2(:,2:end).*sigmoidGradient(z2);
    del2 = del2';
    
%     STEP 4
    Theta2_grad = Theta2_grad + (del3'*a2);  % size(Theta2_grad) -> 10 x 26
    Theta1_grad = Theta1_grad + (del2*a1); % size(Theta2_grad) -> 25 x 401
end

%     STEP 5
% See in Part 3 below.


% Unroll parameters
% See the very botton
%initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
%grad = [Theta1_grad(:) ; Theta2_grad(:)];

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Remember to not take in account the bias term (column in this case).
sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2));
reg_temp  = lambda/2*m * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

J = J + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

temp2_theta = Theta2;
temp1_theta = Theta1;

temp2_theta(:,1) = 0;
temp1_theta(:,1) = 0;

Theta2_grad = 1/m*Theta2_grad;
Theta1_grad = 1/m*Theta1_grad;

Theta2_grad = Theta2_grad + lambda/m * temp2_theta;
Theta1_grad = Theta1_grad + lambda/m * temp1_theta;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
