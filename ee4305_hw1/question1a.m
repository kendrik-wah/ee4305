X = [0; 0.5];
iter_count = 0;
eta_1 = 1e-3;
eta_2 = 2e-1;
stop = 0;
iters = 1e20;
flag = 1;
i = 1;

x = zeros();
y = zeros();
z = zeros();
errors = zeros();
iterations = zeros();

while (flag && i <= iters)
    
    x(i) = X(1);
    y(i) = X(2);
    
    X_prev = X;
    g = rosenGrad(X(1), X(2));
    X = X - (eta_1 * g); % can change eta_1 with eta_2 whenever you want.
    error = norm(X - X_prev);
    
    errors(i) = error;
    iterations(i) = i;
    
    if (stop >= error)
        flag = 0;
    end
    
    i = i + 1;
end

figure;
x_values = [-2:0.01:2]; y_values = [-2:0.01:2];
f = @(x,y) (1-x).^2 + 100*(y-x.^2).^2;
[xx, yy] = meshgrid(x_values, y_values);
ff = f(xx,yy);
fn_output = f(x, y);
contour(xx, yy, ff, 100);

hold on;
scatter(x, y);
xlabel('x');
ylabel('y');
hold off;

figure;
plot(iterations, x);
hold on;
plot(iterations, y);
legend({"x", "y"}, 'Location', 'southeast');
hold off;

figure;
plot(iterations, fn_output);
xlabel('iterations');
ylabel('f');


function val = rosen(x, y)
    val = (1-x)^2 + (100 * (y - (x^2))^2);
end

function Df = rosenGrad(x, y)
    k = [x-1; y-(x^2)];
    Df = [2*k(1)-(400*x*k(2)); 200*k(2)];
end

% when eta = 0.001, 72594 iterations are needed.
% when eta = 0.02, the gradient descent optimization never ends.
