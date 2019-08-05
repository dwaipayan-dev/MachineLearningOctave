c = num_labels;
j_cost = [];
for k = 1:c
  y_k = y==c;
  Z1_k = X * (Theta1)' 
  a1_k = sigmoid(Z1_k);
  a1_k = [ones(m,1);Z1_k];
  Z2_k = a1_k * (Theta2)';
  a2_k = sigmoid(Z2_k);
  H_t_k = max(a2_k, [], 2)
  j_cost(i, 1) = -(y_k .* log(H_t_k) - (1-y_k) .* log(1-H_t_k))/m;
  
endfor
J = sum(j_cost);
t1 = Theta1.^2;
t2 = Theta2.^2;
t1(:,1) = 0;
t2(:,1) = 0;
reg = lambda*(t1+t2)/(2*m);
J = J+reg;