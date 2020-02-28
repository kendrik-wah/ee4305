X = [1 0.5; 1 1.5; 1 3.0; 1 4.0; 1 5.0];
x = [0.5; 1.5; 3.0; 4.0; 5.0];
d = [8.0; 6.0; 5.0; 2.0; 0.5];
y_plot_lls = zeros([1 5]);

% ===== Linear Least Squares ===== %

w_LLS = inv(X' * X) * X' * d;

for i=1:length(x)
   y_plot_lls(i) = w_LLS(2,1)*x(i) + w_LLS(1,1);
end

% ================================ %

% ===== Least Mean Squares ===== %

n_epochs = 100;
epochs = (0:1:99);
w_LMS = [4;2];
y_plot_lms = zeros([1 5]);
eta = 0.2;
bias = zeros([1 100]);
w0 = zeros([1 100]);
error = zeros([1 100]);

for j=1:n_epochs
    
    E = 0;
    
    for k=1:length(x)
        y = X(k,:) * w_LMS;
        e = d(k) - y;
        w_LMS = w_LMS + (eta*X(k,:)'*e);
        
        E = E + e;
    end
    
    E = E/length(x);
    
    bias(j) = w_LMS(1);
    w0(j) = w_LMS(2);
    error(j) = E;
    
    for k=1:length(x)
       y_plot_lms(k) = X(k,:) * w_LMS; 
    end
end

% =============================== %

scatter(x, d);
hold on;
plot(x, y_plot_lls, x, y_plot_lms);
hold off;
legend({'points', 'LLS', 'LMS'}, 'Location', 'northeast');

figure;

plot(epochs, bias, epochs, w0);
legend({'bias', 'w0'}, 'Location', 'northeast');

figure;

plot(epochs, error);
legend({'error'}, 'Location', 'northeast');
