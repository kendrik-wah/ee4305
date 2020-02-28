OR_x = [1 0 0; 1 0 1; 1 1 0; 1 1 1];
OR_x1 = [0;0;1;1];
OR_x2 = [0;1;0;1];
OR_d = [0;1;1;1];
OR_w = [rand(); rand(); rand()];
OR_eta = 1;

iters = (0:1:99);
w_0_vals = zeros([1 20]);
w_1_vals = zeros([1 20]);
w_2_vals = zeros([1 20]);
i = 1;
flag = 1;

w_0_vals(i) = OR_w(1);
w_1_vals(i) = OR_w(2);
w_2_vals(i) = OR_w(3);

while (flag)
    
    j = 1;
    flag = 0;
    
    while (j <= length(OR_d))
        
        OR_y = (OR_x(j,:) * OR_w)>=0; % the mid of this equation calculates the value of v. >= 0 is the activation function.
        OR_e = OR_d(j) - OR_y;        
        if (OR_e ~= 0)
            
            w_0_vals(i) = OR_w(1);
            w_1_vals(i) = OR_w(2);
            w_2_vals(i) = OR_w(3);
            OR_w = OR_w + (OR_eta * (OR_e' * OR_x(j,:))');
            
            i = i + 1;
            flag = 1;
        end
        
        j = j + 1;
    end
end

w_0_vals(i) = OR_w(1);
w_1_vals(i) = OR_w(2);
w_2_vals(i) = OR_w(3);

x_point = (0:0.1:1.4);

y_point = zeros([1 15]);
k = 1;

while (k <= 15)
   y_point(k) = -(OR_w(2)/OR_w(3))*x_point(k) - (OR_w(1)/OR_w(3));
   k = k + 1;
end

scatter(OR_x1,OR_x2);
xlabel('x1');
ylabel('x2');
hold on;
plot(x_point, y_point);
hold off;

figure;

plot(iters(1:i), w_0_vals(1:i), iters(1:i), w_1_vals(1:i), iters(1:i), w_2_vals(1:i));
xlabel('iterations');
ylabel('weight values');
legend({'b','w0','w1'}, 'Location', 'northeast');