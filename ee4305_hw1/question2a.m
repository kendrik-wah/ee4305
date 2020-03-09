clear;

train_delta = 5e-2;
val_delta = 1e-2;
test_delta = 1e-2;
min = -1;
max = 1;
test_min = -3;
test_max = 3;
epochs = 5e2;
hidden_neurons = [1,2,3,4,5,6,7,8,9,10,20,50,100];
epoch_arr = [1:1:epochs];
training_fns = ['traingd', 'traingda', 'traingdx', 'trainlm', 'trainbr'];
eta = 0.005; % to be experimented upon

% Training data
train_X = [min: train_delta: max];
train_X_label = fnTrigo(train_X);

% Validation data
val_X = [min: val_delta: max];
val_X_ans = fnTrigo(val_X);

% Test data
test_X = [test_min: test_delta: test_max];
test_X_ans = fnTrigo(test_X);

% Training, validating and testing the neural network with different number
% of hidden neurons. Everything is run in the same loop in order to prevent
% different values of input, layer weights and biases instantiated when
% train_seq is called.
for i = 1: length(hidden_neurons)
    
    % Train the neural network.
    net = train_seq(hidden_neurons(i), train_X, train_X_label, epochs, eta);
    val_train = net(val_X);
    test_train = net(test_X);
    
    filename_plot_val = sprintf("q2a_change_hidden\\plot_hidden_%d", hidden_neurons(i));
    filename_scatter_val = sprintf("q2a_change_hidden\\scatter_hidden_%d", hidden_neurons(i));
    filename_plot_test = sprintf("q2a_extrapolate\\plot_hidden_%d", hidden_neurons(i));
    filename_scatter_test = sprintf("q2a_extrapolate\\scatter_hidden_%d", hidden_neurons(i));
    
    % plot validation data
    plot(val_X, val_X_ans, val_X, val_train);
    xlabel('x');
    ylabel('y');
    legend({'actual', 'predicted'}, 'Location', 'northwest');
    saveas(gcf, filename_plot_val, 'png');
    
    plot(val_X, val_X_ans);
    hold on;
    scatter(val_X, val_train, '.');
    legend({'actual', 'predicted'}, 'Location', 'northwest');
    saveas(gcf, filename_scatter_val, 'png');
    hold off;
    
    % plot test data  
    plot(test_X, test_X_ans, test_X, test_train);
    xlabel('x');
    ylabel('y');
    legend({'actual', 'predicted'}, 'Location', 'northwest');
    saveas(gcf, filename_plot_test, 'png');
    
    plot(test_X, test_X_ans);
    hold on;
    scatter(test_X, test_train, '.');
    legend({'actual', 'predicted'}, 'Location', 'northwest');
    saveas(gcf, filename_scatter_test, 'png');
    hold off;
end

function val = fnTrigo(x)
    val = 1.2*sin(pi*x) - cos(2.4*pi*x);
end

function [net, val_train, test_train] = train_seq(n, train_X, train_X_output, epochs, eta)

    x_train = num2cell(train_X);
    x_train_label = num2cell(train_X_output);
    
    display("number of hidden neurons:", num2str(n)); % for message passing
    display("number of epochs:", num2str(epochs));    % for message passing
    
    net = fitnet(n, 'traingda'); % traingda appears to give the best outcomes.
    
    net.divideParam.trainRatio = 1;
    net.divideParam.valRatio = 0;
    net.divideParam.testRatio = 0;
    net.layers{1}.transferFcn = 'tansig'; % to be experimented upon
    net.layers{2}.transferFcn = 'purelin'; % to be experimented upon
    net.trainParam.lr = eta;
    
    for i=1:epochs
        idx = randperm(length(x_train));
        net = adapt(net, x_train(:, idx), x_train_label(:, idx));      
    end
end