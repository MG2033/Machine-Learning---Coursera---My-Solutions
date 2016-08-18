function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


gamma = (X*Theta'-Y).^2 .* R;

J = (1/2)*(gamma*ones(num_users,1))'*ones(num_movies,1);                %gamma is just a dummy name, ones is used for vectorized summation!

%Add regularization
reg1 = 0; reg2=0;
for j=1:num_users
    reg1 = reg1 + Theta(j,:)*Theta(j,:)';                                     %Theta(j,:) is a row vector.
end
reg1 = reg1 * (lambda/2);
for i=1:num_movies
    reg2 = reg2 + X(i,:)*X(i,:)';
end
reg2 = reg2 * (lambda/2);

J = J + reg1 + reg2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Gradient Computation

for i=1:num_movies
    X_grad(i,:) = ((Theta*X(i,:)'-Y(i,:)').*R(i,:)')'*Theta + lambda * X(i,:);          %Very difficult!!
end

for j=1:num_users
    Theta_grad(j,:) = (((X*Theta(j,:)')-Y(:,j)).*R(:,j))'*X + lambda * Theta(j,:);
end

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
