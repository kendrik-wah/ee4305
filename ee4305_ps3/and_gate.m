X = [1 0 0; 1 0 1; 1 1 0; 1 1 1];
x1 = [0;0;1;1];
x2 = [0;1;0;1];
d = [0;0;0;1];
w = [rand(); rand(); rand()];
eta = 1.0;

iters = (0:1:99);
w_0_vals = zeros([1 100]);
w_1_vals = zeros([1 100]);
w_2_vals = zeros([1 100]);
i = 1;
flag = 1;

w_0_vals(i) = w(1);
w_1_vals(i) = w(2);
w_2_vals(i) = w(3);

while (flag)
    
    flag = 0; 
    
    for j=1:length(d)
        
        y = (X(j,:) * w)>=0; % >= 0 is the activation function.
        e = d(j) - y;   
        if (e ~= 0)
            
            w_0_vals(i) = w(1);
            w_1_vals(i) = w(2);
            w_2_vals(i) = w(3);
            
            w = w + (eta * (e * X(j,:)'));
            flag = 1;
            i = i + 1;
        end
                    
    end
end

w_0_vals(i) = w(1);
w_1_vals(i) = w(2);
w_2_vals(i) = w(3);

x_point = (0:0.1:1.4);
y_point = zeros([1 15]);
k = 1;

while (k <= 15)
   y_point(k) = -(w(2)/w(3))*x_point(k) - (w(1)/w(3));
   k = k + 1;
end

scatter(x1,x2);
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