clear;

train_delta = 5e-2;
val_delta = 1e-2;
min = -1;
max = 1;
test_min = -3;
test_max = 3;
epochs = 1e2;
n = 6;
hidden_neurons = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100];
epoch_arr = [1:1:epochs];
eta = 0.01;

% Training data
train_X = [min: train_delta: max];
train_X_label = fnTrigo(train_X);

% Validation data
val_X = [min: val_delta: max];
val_X_ans = fnTrigo(val_X);

net = fitnet(n, 'trainlm');
net.trainParam.epochs = epochs;
net.inputWeights{1,1}.learnParam.lr = eta;
net.layerWeights{2,1}.learnParam.lr = eta;
net.biases{1}.learnParam.lr = eta;
net.biases{2}.learnParam.lr = eta;

[net, e, a] = train(net, train_X, train_X_label);

function val = fnTrigo(x)
    val = 1.2*sin(pi*x) - cos(2.4*pi*x);
end