m=size(X,1);
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

XX=[ones(m, 1) X];

XXval =[ones(size(Xval, 1), 1) Xval];

for i =1:m;
      
 [theta] = trainLinearReg([ones(i, 1) X(1:i,:)], y(1:i), lambda);
 
 error_train(i)= 1./(2*i).*sum((XX(1:i,:)*theta-y(1:i)).^2);
 error_val(i)= 1./(2*size(yval,1)).*sum((XXval*theta-yval).^2);
 
 
end