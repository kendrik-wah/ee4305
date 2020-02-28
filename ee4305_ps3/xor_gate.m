XOR_x = [1 0 0; 1 0 1; 1 1 0; 1 1 1];
XOR_x1 = [0;0;1;1];
XOR_x2 = [0;1;0;1];
XOR_d = [0;1;1;0];
XOR_w = [rand(); rand(); rand()];
XOR_eta = 1;

iters = (0:1:99);
w_0_vals = zeros([1 100]);
w_1_vals = zeros([1 100]);
w_2_vals = zeros([1 100]);
i = 1;
flag = 1;

w_0_vals(i) = XOR_w(1);
w_1_vals(i) = XOR_w(2);
w_2_vals(i) = XOR_w(3);
    
while (flag & i <100)
    
    flag = 0;
    
    for (j=1:length(XOR_d))
        
        XOR_y = (XOR_x(j,:) * XOR_w)>=0; % the mid of this equation calculates the value of v. >= 0 is the activation function.
        XOR_e = XOR_d(j) - XOR_y;        
        if (XOR_e ~= 0)
            
            w_0_vals(i) = XOR_w(1);
            w_1_vals(i) = XOR_w(2);
            w_2_vals(i) = XOR_w(3);
            XOR_w = XOR_w + (XOR_eta * (XOR_e' * XOR_x(j,:))');
            
            i = i+1;
            flag = 1;
        end
    end
end

plot(iters, w_0_vals, iters, w_1_vals, iters, w_2_vals);
xlabel('iterations');
ylabel('weight values');
legend({'b','w0','w1'}, 'Location', 'northeast');