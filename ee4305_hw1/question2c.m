clear;

train_delta = 5e-2;
test_delta = 1e-2;
val_delta = 1e-2;
min = -1;
max = 1;
test_min = -3;
test_max = 3;
epochs = 5e2;
n = 6;
hidden_neurons = [1,2,3,4,5,6,7,8,9,10,20,50];
epoch_arr = [1:1:epochs];
eta = 0.01;

% Training data
train_X = [min: train_delta: max];
train_X_label = fnTrigo(train_X);

% Validation data
val_X = [min: val_delta: max];
val_X_ans = fnTrigo(val_X);

% Test data
test_X = [test_min: test_delta: test_max];
test_X_ans = fnTrigo(test_X);

% Change the number of hidden neurons and run batch training on it.
for i=1:length(hidden_neurons)
    
    % Train the neural network.
    [net, tr] = train_bat(hidden_neurons(i), train_X, train_X_label, epochs, eta);
    
    % Validate the neural network.
    val_X_net_output = sim(net, val_X);
    
    % Test the neural network.
    test_X_net_output = sim(net, test_X);
    
    filename_scatter_val = sprintf("q2c_change_hidden\\scatter_hidden_%d", hidden_neurons(i));
    filename_plot_val = sprintf("q2c_change_hidden\\plot_hidden_%d", hidden_neurons(i));
    filename_scatter_test = sprintf("q2c_extrapolate\\scatter_hidden_%d", hidden_neurons(i));
    filename_plot_test = sprintf("q2c_extrapolate\\plot_hidden_%d", hidden_neurons(i));
    
    figure;
    plotperform(tr);
    
    figure
    % plot validation data
    scatter(val_X, val_X_ans, 'x');
    hold on;
    scatter(val_X, val_X_net_output, 'x');
    hold off;
    xlabel('x');
    ylabel('y');
    legend({'actual', 'predicted'}, 'Location', 'northwest');
    saveas(gcf, filename_scatter_val, 'png');
    
    plot(val_X, val_X_ans, val_X, val_X_net_output);
    xlabel('x');
    ylabel('y');
    legend({'actual', 'predicted'}, 'Location', 'northwest');
    saveas(gcf, filename_plot_val, 'png');
    
    % plot test data
    scatter(test_X, test_X_ans, 'x');
    hold on;
    scatter(test_X, test_X_net_output, 'x');
    hold off;
    xlabel('x');
    ylabel('y');
    legend({'actual', 'predicted'}, 'Location', 'northwest');
    saveas(gcf, filename_scatter_test, 'png');
    
    plot(test_X, test_X_ans, test_X, test_X_net_output);
    xlabel('x');
    ylabel('y');
    legend({'actual', 'predicted'}, 'Location', 'northwest');
    saveas(gcf, filename_plot_test, 'png');
end

function val = fnTrigo(x)
    val = 1.2*sin(pi*x) - cos(2.4*pi*x);
end

function [net, tr] = train_bat(n, train_X, train_X_label, epochs, eta)

    net = fitnet(n, 'trainbr');
    net.trainParam.epochs = epochs;
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'purelin';
    net.inputWeights{1,1}.learnParam.lr = eta;
    net.layerWeights{2,1}.learnParam.lr = eta;
    net.biases{1}.learnParam.lr = eta;
    net.biases{2}.learnParam.lr = eta;
    
    [net, tr] = train(net, train_X, train_X_label);
end