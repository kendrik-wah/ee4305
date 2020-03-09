clear;

% train has 501 items. val has 167 items.
n_train = 501;
n_val = 167;
eta = 0.01;
epochs = 8e2;
training_sets = [1:1:501];

% Defining training data
filepath_train = "group_3\\train";
train_folder = dir(filepath_train);

% Retrieve validation data
filepath_val = "group_3\\val";
val_folder = dir(filepath_val);

% redefining the data sets according to scales, full clarity
[training_data, training_label] = extract_img_set(filepath_train, train_folder, n_train, 1);
[validation_data, validation_label] = extract_img_set(filepath_val, val_folder, n_val, 1);
epoch = [1:1:epochs];

n_components = extract_component_numbers(filepath_train, train_folder, n_train, 256);

dimensions_to_test = [1, ceil(n_components/20), ceil(n_components/10), ceil(n_components/5), n_components, min(n_components+50), 255];
display(dimensions_to_test);

% change number of components, given a certain scale.
for k=1:length(dimensions_to_test)
   filename = sprintf("q3d\\sequential_epoch_%d_components", dimensions_to_test(k));
   [net, accu_train_pca, accu_val_pca] = train_seq(training_data, training_label, validation_data, validation_label, epochs, filename, dimensions_to_test(k));
end






function [net, accu_train, accu_val] = train_seq(training_data, training_label, validation_data, validation_label, epochs, filename, comp)
    % define patternnet
    net = patternnet(comp);
    net.trainFcn = 'trainrp';
    net.trainParam.lr = 0.01;
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'logsig';
    net.trainParam.epochs=epochs;
    epoch = [1:1:epochs];

    accu_train = zeros(1, epochs);
    accu_val = zeros(1, epochs);

    % Training loop
    for i=1:epochs
        idx = randperm(size(training_data, 2));
        net = adapt(net, training_data(:,idx), training_label(:,idx));

        pred_train = net(training_data(:,idx));
        accu_train(i) = 1 - mean(abs(pred_train - training_label(:,idx)));
        
        val_train = net(validation_data);
        accu_val(i) = 1 - mean(abs(val_train - validation_label));
    end
    
    plot(epoch, accu_train, epoch, accu_val);
    xlabel("epoch");
    ylabel("accuracy (%)");
    legend({'training', 'validation'}, 'Location', 'southeast');
    saveas(gcf, filename, 'png');
end

function [img, label] = extract_img(filepath, folder, i, scale)
% Extracts the i-th image and its corresponding label as denoted in the given filepath. Only one
% image is extracted at a time.
    filename = filepath + '\\' + folder(i).name;
    img = imread(filename);
    img = imresize(img, scale);
    img = img(:);
    tmp = strsplit(filename, {'_', '.'});
    label = str2num(tmp{3});
   
end

function [img_set, label_set] = extract_img_set(filepath, folder, count, scale)
    
    sz = (256*scale)^2;
    img_set = zeros([sz count]);
    label_set = zeros([1 count]);
    for i=3:count+2
        [img, label] = extract_img(filepath, folder, i, scale);
        img_set(:,i-2) = img;
        label_set(:,i-2) = label;
    end
    
end

function n_components = extract_component_numbers(filepath, folder, count, dimn)
    effective_ranks = zeros([1 count]);
    for i=3:count+2
        I = imread(filepath + '\\' + folder(i).name);
        singular_val = svd(double(I));
        sv_sum = 0;
        k_sv_sum = 0;
        for j = 1:dimn
            sv_sum = sv_sum + singular_val(j);
        end

        for k = 1:dimn
            k_sv_sum = k_sv_sum + singular_val(k);
            if k_sv_sum/sv_sum >= 0.99
                effective_ranks(i) = k;
                break
            end
        end
    end
    n_components = ceil(mean(effective_ranks));
end