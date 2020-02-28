COMP_x = [1 0; 1 1];
COMP_x1 = [0;1];
COMP_d = [1;0];
COMP_w = [0.3; 0.6];
COMP_eta = 1;

iters = (0:1:99);
w_0_vals = zeros([1 100]);
w_1_vals = zeros([1 100]);
i = 1;
flag = 1;

w_0_vals(i) = COMP_w(1);
w_1_vals(i) = COMP_w(2);

while (flag)
    
    j = 1;
    flag = 0;
    
    while (j <= length(COMP_d))
        
        COMP_y = (COMP_x(j,:) * COMP_w)>=0; % the mid of this equation calculates the value of v. >= 0 is the activation function.
        COMP_e = COMP_d(j) - COMP_y;        
        if (COMP_e ~= 0)
            
            w_0_vals(i) = COMP_w(1);
            w_1_vals(i) = COMP_w(2);
            COMP_w = COMP_w + (COMP_eta * (COMP_e' * COMP_x(j,:))');
            
            i = i + 1;
            flag = 1;
        end
        
        j = j + 1;
    end
end

x_point = (0:0.1:1.4);

w_0_vals(i) = COMP_w(1);
w_1_vals(i) = COMP_w(2);

y_point = zeros([1 15]);
k = 1;

while (k <= 15)
   y_point(k) = -(COMP_w(1)/COMP_w(2));
   k = k + 1;
end

scatter(COMP_x1,[0;0]);
xlabel('x1');
ylabel('x2');
hold on;
plot(y_point, x_point);
hold off;

figure;

plot(iters(1:i), w_0_vals(1:i), iters(1:i), w_1_vals(1:i));
xlabel('iterations');
ylabel('weight values');
legend({'b','w0'}, 'Location', 'northeast');