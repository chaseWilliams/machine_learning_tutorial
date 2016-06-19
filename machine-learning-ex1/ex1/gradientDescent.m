function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

  h = theta(1) + X(:,2) * theta(2);
  sigma = sum(h - y);
  temp0 = theta(1) - alpha * (1/m) * sigma;
  h = theta(1) + X(:, 2) * theta(2);
  sigma = sum((h - y) .* X(:,2));
  temp1 = theta(2) - alpha * (1/m) * sigma;

  if ((temp0 == theta(1)) == 1)
    if ((temp1 == theta(2)) == 1)
      break;
    end
  end

  theta = [temp0; temp1];
  
  % Save the cost J in every iteration
  J_history(iter) = computeCost(X, y, theta);
end
disp('local optimum reached- theta is');
disp(theta);
end
