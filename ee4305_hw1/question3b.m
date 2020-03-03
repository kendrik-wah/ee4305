clear;

% train has 501 items. val has 167 items.
n_train = 501;
n_val = 167;
eta = 0.01;
epoch_test = [1e2, 2e2, 5e2, 1e3, 1.25e3];
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





% Training loop
for i=1:length(epoch_test)
    
   epochs = epoch_test(i);
   epoch = [1:1:epochs]; 
   [net, accu_train, accu_val] = train_seq(training_data, training_label, validation_data, validation_labe, epochs);
   
    filename = sprintf("q3b_sequential\\sequential_epoch_%d", epochs);
    plot(epoch, accu_train, epoch, accu_val);
    xlabel("epoch");
    ylabel("accuracy (%)");
    legend({'training', 'validation'}, 'Location', 'northwest');
    saveas(gcf, filename, 'png');
end






function [net, accu_train, accu_val] = train_seq(training_data, training_label, validation_data, validation_label, epochs)
    % define perceptron
    net = perceptron;
    net.trainParam.epochs = epochs;

    accu_train = zeros(1, epochs);
    accu_val = zeros(1, epochs);

% Training loop
    for i=1:epochs
        idx = randperm(size(training_data, 2));
        [net,a,e] = adapt(net, training_data(:,idx), training_label(:,idx));

        pred_train = net(training_data(:,idx));
        accu_train(i) = 1 - mean(abs(pred_train - training_label(:,idx)));

        val_train = net(validation_data);
        accu_val(i) = 1 - mean(abs(val_train - validation_label));
    end
end

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