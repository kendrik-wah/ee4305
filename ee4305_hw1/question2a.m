clear;

train_delta = 5e-2;
val_delta = 1e-2;
min = -1;
max = 1;
test_min = -3;
test_max = 3;
epochs = 1e2;
hidden_neurons = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100];
epoch_arr = [1:1:epochs];
eta = 0.01;

% Training data
train_X = [min: train_delta: max];
train_X_label = fnTrigo(train_X);

% Validation data
val_X = [min: val_delta: max];
val_X_ans = fnTrigo(val_X);

% To determine the best number of hidden layers used.
for i = 1: length(hidden_neurons)
    
    % Train the neural network first.
    [net, a, e] = train_seq(hidden_neurons(i), train_X, train_X_label, epochs, eta);
    
    % Test the neural network on the validation data.
    val_X_net_output = sim(net, val_X);
    
    filename_scatter = sprintf("C:\\Users\\kendrik\\Documents\\MATLAB\\ee4305\\ee4305_hw1\\q2a_change_hidden\\scatter_hidden_%d", hidden_neurons(i));
    filename_plot = sprintf("C:\\Users\\kendrik\\Documents\\MATLAB\\ee4305\\ee4305_hw1\\q2a_change_hidden\\plot_hidden_%d", hidden_neurons(i));

    % plot validation data
    scatter(val_X, val_X_ans, 'x');
    hold on;
    scatter(val_X, val_X_net_output, 'x');
    hold off;
    xlabel('x');
    ylabel('y');
    legend({'actual', 'predicted'}, 'Location', 'northwest');
    saveas(gcf, filename_scatter, 'png');

    plot(val_X, val_X_ans, val_X, val_X_net_output);
    xlabel('x');
    ylabel('y');
    legend({'actual', 'predicted'}, 'Location', 'northwest');
    saveas(gcf, filename_plot, 'png');
end

% Based on the loop above, it appears that 1-6-1 is the best configuration.
% This coincides with the idea that by minimising the number of line
% segments and then gradually adding the number of hidden neurons, a more
% accurate neural network can be obtained for function approximation.
% To further show this, extrapolate the range of x: -3 <= x <= 3

for i=1:length(hidden_neurons)
    test_delta = val_delta;
    test_x = [test_min:test_delta:test_max];
    test_y_true = fnTrigo(test_x);
    [net, a, e] = train_seq(hidden_neurons(i), train_X, train_X_label, epochs, eta);
    test_y_calc = sim(net, test_x);
    
    filename_scatter = sprintf("C:\\Users\\kendrik\\Documents\\MATLAB\\ee4305\\ee4305_hw1\\q2a_extrapolate\\scatter_hidden_%d", hidden_neurons(i));
    filename_plot = sprintf("C:\\Users\\kendrik\\Documents\\MATLAB\\ee4305\\ee4305_hw1\\q2a_extrapolate\\plot_hidden_%d", hidden_neurons(i));
    
    scatter(test_x, test_y_true, 'x');
    hold on;
    scatter(test_x, test_y_calc, 'x');
    hold off;
    xlabel('x');
    ylabel('y');
    legend({'actual', 'predicted'}, 'Location', 'northwest');
    saveas(gcf, filename_scatter, 'png');
    
    plot(test_x, test_y_true, test_x, test_y_calc);
    xlabel('x');
    ylabel('y');
    legend({'actual', 'predicted'}, 'Location', 'northwest');
    saveas(gcf, filename_plot, 'png');
end

function val = fnTrigo(x)
    val = 1.2*sin(pi*x) - cos(2.4*pi*x);
end

function [net, a, e] = train_seq(n, train_X, train_X_output, epochs, eta)

    x_train = num2cell(train_X);
    x_train_label = num2cell(train_X_output);
    
    net = fitnet(n);
    
    net.divideParam.trainRatio = 1;
    net.divideParam.valRatio = 0;
    net.divideParam.testRatio = 0;
    net.inputWeights{1,1}.learnParam.lr = eta;
    net.layerWeights{2,1}.learnParam.lr = eta;
    net.biases{1}.learnParam.lr = eta;
    net.biases{2}.learnParam.lr = eta;
    
    for i=1:epochs
        idx = randperm(length(x_train));
        [net, a, e] = adapt(net, x_train(:, idx), x_train_label(:, idx));
    end
end