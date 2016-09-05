function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 
% =========================================================================
% p = 8;
% % Map X onto Polynomial Features and Normalize
% X_poly = polyFeatures(X, p);
% =========================================================================
for iter = 1:(size(X,1))
    for p_iter = 1:p
       X_poly(iter, p_iter) = X(iter).^p_iter; 
    end
end
