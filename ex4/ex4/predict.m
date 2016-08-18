function [a3 z2 a2 z3] = predict(Theta1, Theta2, X)

% MG: This is a modified version of the function. I used it to return the probability vector in order to be able to calculate the cost correctly using nnCostFunction.
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

z2 = Theta1*X;
a2 = sigmoid(z2); a2 = [1; a2];
z3 = Theta2*a2;
a3 = sigmoid(z3);
[dummy, index] = max(a3, [], 2);


% =========================================================================


end
