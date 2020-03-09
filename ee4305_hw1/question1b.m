X = [0; 0.5];
iter_count = 0;
eta = 1e-3;
stop = 0;
iters = 1e20;
flag = 1;
i = 1;

x = zeros();
y = zeros();
z = zeros();
errors = zeros();
iterations = [];

while (flag && i <= iters)
    
    x(i) = X(1);
    y(i) = X(2);
    
    X_prev = X;
    del_X = -inv(rosenHess(X(1), X(2))) * rosenGrad(X(1), X(2));
    X = X + del_X;
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
contour(xx, yy, ff, 100);

hold on;
plot(x, y, 'r');
xlabel('x');
ylabel('y');

for i=1:length(x)
   z(i) = rosen(x(i), y(i)); 
end

figure;
contour3(xx, yy, ff, 100);
hold on;
plot3(x, y, z, 'r');
xlabel('x');
ylabel('y');
zlabel('z');

figure;
plot(iterations, x, iterations, y);
xlabel('iterations');
ylabel('value');
legend({'x', 'y'}, 'Location', 'northeast');

figure;
plot(x, y);
xlabel('x');
ylabel('y');

figure;
plot(iterations, z);
xlabel('iterations');
ylabel('f');

function val = rosen(x, y)
    val = (1-x)^2 + (100 * (y - (x^2))^2);
end

function Df = rosenGrad(x, y)
    k = [x-1; y-(x^2)];
    Df = [2*k(1)-(400*x*k(2)); 200*k(2)];
end

function H = rosenHess(x, y)
    df2dx2 = 2 - 400*y + 1200*(x^2);
    df2dy2 = 200
    df2dxdy = -400 * x;
    H = [df2dx2, df2dxdy; df2dxdy, df2dy2];
end