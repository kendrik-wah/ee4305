NAND_x = [1 0 0; 1 0 1; 1 1 0; 1 1 1];
NAND_x1 = [0;0;1;1];
NAND_x2 = [0;1;0;1];
NAND_d = [1;1;1;0];
NAND_w = [rand(); rand(); rand()];
NAND_eta = 1;

iters = (0:1:99);
w_0_vals = zeros([1 20]);
w_1_vals = zeros([1 20]);
w_2_vals = zeros([1 20]);
i = 1;
flag = 1;

w_0_vals(i) = NAND_w(1);
w_1_vals(i) = NAND_w(2);
w_2_vals(i) = NAND_w(3);

while (flag)
    
    j = 1;
    flag = 0;
    
    while (j <= length(NAND_d))
        
        NAND_y = (NAND_x(j,:) * NAND_w)>=0; % the mid of this equation calculates the value of v. >= 0 is the activation function.
        NAND_e = NAND_d(j) - NAND_y;        
        if (NAND_e ~= 0)
            
            w_0_vals(i) = NAND_w(1);
            w_1_vals(i) = NAND_w(2);
            w_2_vals(i) = NAND_w(3);
            
            NAND_w = NAND_w + (NAND_eta * (NAND_e' * NAND_x(j,:))');
            
            i = i + 1;
            flag = 1;
        end
        
        j = j + 1;
    end
end

w_0_vals(i) = NAND_w(1);
w_1_vals(i) = NAND_w(2);
w_2_vals(i) = NAND_w(3);

x_point = (0:0.1:1.4);

y_point = zeros([1 15]);
k = 1;

while (k <= 15)
   y_point(k) = -(NAND_w(2)/NAND_w(3))*x_point(k) - (NAND_w(1)/NAND_w(3));
   k = k + 1;
end

scatter(NAND_x1,NAND_x2);
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