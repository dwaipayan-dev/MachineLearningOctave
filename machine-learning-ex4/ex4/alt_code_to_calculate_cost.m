y_k = (y==num_labels);
size(y_k)
X = [ones(m,1), X];
size(X)
size(Theta1);
  for k = 1:num_labels
    y_k = (y == k);
    Z1_k = X*Theta1';
    a1_k = sigmoid(Z1_k);
    a1_k = [ones(m,1), a1_k];
    Z2_k = a1_k*Theta2';
    a2_k = sigmoid(Z2_k);
    H_k = a2_k(:,k);
    cost = -y_k.*log(H_k) - (1-y_k).*log(1-H_k);
    j_cost = sum(cost)/m;
    J = J+ j_cost;
  endfor