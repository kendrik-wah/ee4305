clear;

% train has 501 items. val has 167 items.
n_train = 501;
n_val = 167;
eta = 0.005;
epoch_test = [1:1:100];
val_acc = zeros([1 length(epoch_test)]);

epoch = [1:1:epochs];
training_sets = [1:1:501];

% Defining training data
filepath_train = "group_3\\train";
train_folder = dir(filepath_train);
training_data = zeros([65536, n_train]);
training_label = zeros([1, n_train]);

% Extract training data
for i=3:n_train+2
    [img, label] = extract_img(filepath_train, train_folder, i);
    training_data(:,i-2) = img;
    training_label(:,i-2) = label;
end

% Retrieve validation data
filepath_val = "group_3\\val";
val_folder = dir(filepath_val);
validation_data = zeros([65536, n_val]);
validation_label = zeros([1, n_val]);

% Extract validation data
for i=3:n_val+2
    [img, label] = extract_img(filepath_val, val_folder, i);
    validation_data(:,i-2) = img;
    validation_label(:,i-2) = label;
end





for i=1:length(epoch_test)
    
    net = perceptron;
    epochs = epoch_test(i);
    net.trainParam.epochs=epochs;
    accu_train = zeros(1, epochs);
    accu_val = zeros(1, epochs);
    
    net = train(net, training_data, training_label);
    
    train_count = [1:1:n_train];
    train_out = net(training_data);

    val_count = [1:1:n_val];
    val_out = net(validation_data);
    
    accuracy = 1- mean(abs(val_out - validation_label)); 
    val_acc(i) = accuracy;
end

filename = sprintf("q3b_batch\\batch_accuracy");
plot(epoch_test, val_acc);
xlabel("epochs");
ylabel("accuracy");
legend({"accuracy"}, 'Location', 'northeast');
saveas(gcf, filename, 'png');







function [img, label] = extract_img(filepath, folder, i)
% Extracts the i-th image and its corresponding label as denoted in the given filepath. Only one
% image is extracted at a time.
    filename = filepath + '\\' + folder(i).name;
    img = imread(filename);
    img = img(:);
    tmp = strsplit(filename, {'_', '.'});
    label = str2num(tmp{3});
   
end

function [old_dim, old_img, new_dim, new_img] = resize_img(filepath, folder, i, scale)
    filename = filepath + '\\' + folder(i).name;
    old_img = imread(filename);
    old_dim = size(old_img);
    new_img = imresize(old_img, scale);
    new_dim = size(new_img);
end

function [old_dim, old_img, coeff, img, x_form] = pca_img(filepath, folder, i, nComp)
    filename = filepath + '\\' + folder(i).name;
    old_img = imread(filename);
    old_dim = size(old_img);
    dbl_img = double(old_img);
    dbl_img_mean = mean(dbl_img);
    dbl_img_adjusted = dbl_img-dbl_img_mean;
    
    [coeff, score] = pca(dbl_img_adjusted);
    x_form = score(:,1:nComp)*coeff(:,1:nComp)';
    x_form = x_form + dbl_img_mean;
    x_form = uint8(x_form);
    img = dbl_img;
end

% Possible improvements that can be made:
% 1) Change performFcn and see what happens.
% 2) Change number of epochs to below 66 and then see the overall error.