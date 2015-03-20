function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta




for i=1:m;
    c =-theta'*X(i,:)';
    a =-1*y(i)*log(1./(1.+exp(c)));
    b =(1-y(i))*log(1-1./(1.+exp(c)));
    J = J+1./m*(a-b);
end

a=0;
for i=2:size(theta);
    a =a+ lambda./(2*m)*theta(i).^2;
end
J = J+a;
%=============================================================

%vectorized version of J, not part of the assignment; it is part of
%assignment 3

 %J=1./m*sum((-y'.*log(1./(1+exp(-theta'*X'))))-((1-y)'.*log(1-1./(1+exp(-theta'*X')))))+lambda./(2*m)*sum(theta(2:end).^2);


  
for j=1:size(X,2);
       for i=1:m;
           grad(j)=grad(j)+1./m*((1./(1+exp(-(theta'*X(i,:)'))))-y(i)).*X(i,j)^min(j-1,1);
       end
       if j > 1;
       grad(j)=grad(j)+lambda./m*theta(j);
end

end

%vectorized version of grad, not part of the assignment; it is part of
%assignment 3

 %grad=1./m*(1./(1+exp(-theta'*X')) -y')*X+[0,lambda./m*(theta(2:end))'];
