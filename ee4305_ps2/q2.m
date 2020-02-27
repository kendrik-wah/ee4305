w_1 = [0;0];
w_2 = [0;0];
r_xd = [0.8182;0.354];
R_x = [1 0.8182; 0.8182 1];
eta_1 = 0.3;
eta_2 = 1.0;
iterations = 100;
iters = (1:1:100);
weight_bias = zeros([1 100]);
weight_weight = zeros([1 100]);
error_1 = zeros([1 100]);
error_2 = zeros([1 100]);

for i=1:iterations
    
    weight_bias(i) = w_2(1);
    weight_weight(i) = w_2(2);
    
    E_1 = (-r_xd'*w_2)+(0.5*w_2'*R_x*w_2);
    grad_2 = -r_xd'+(w_2'*R_x);
    w_2 = w_2 - (eta_2*grad_2)';
    
    error_1(i) = E_1;
    
end

plot(iters, error_1);
xlabel('iterations');
ylabel('error');
legend({'\eta = 1.0'}, 'Location', 'northeast');

figure;
plot(iters, weight_bias);
xlabel('iterations');
ylabel('bias values');

figure;
plot(iters, weight_weight);
xlabel('iterations');
ylabel('weight values');