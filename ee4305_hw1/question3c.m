clear;

% train has 501 items. val has 167 items.
n_train = 501;
n_val = 167;
eta = 0.01;
epoch_test = [0:20:200];
val_acc = zeros([1 length(epoch_test)]);

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




n_components = extract_component_numbers(filepath_train, train_folder, n_train, 256);

dimensions_to_test = [1, ceil(n_components/20), ceil(n_components/10), ceil(n_components/5), n_components, min(n_components+50), 255];
display(dimensions_to_test);

for h=1:length(dimensions_to_test)

    accu_train = zeros(1, length(epoch_test));
    accu_val = zeros(1, length(epoch_test));

    for i=1:length(epoch_test)

        net = patternnet(dimensions_to_test(h));
        net.trainFcn = 'trainscg';
        net.layers{1}.transferFcn = 'tansig';
        net.layers{2}.transferFcn = 'logsig';
        net.trainParam.lr = eta;
        net.divideFcn = 'dividetrain';
%         net.inputWeights{1,1}.learnParam.lr = eta;
%         net.layerWeights{2,1}.learnParam.lr = eta;
        net.trainParam.epochs=epoch_test(i);

        net = train(net, training_data, training_label);

        train_out = net(training_data);
        acc_train(i) = 1-mean(abs(train_out - training_label));

        val_out = net(validation_data);    
        acc_val(i) = 1- mean(abs(val_out - validation_label)); 
    end
    
    display("Components: ", num2str(dimensions_to_test(h)));
    
    display("training accuracies: ");
    display(acc_train);
    
    display("validation accuracies: ");
    display(acc_val);

    filename = sprintf("q3c\\%d_comp_batch_accuracy", dimensions_to_test(h));
    plot(epoch_test,  acc_train, epoch_test, acc_val);
    xlabel("epochs");
    ylabel("accuracy");
    legend({"training", "validation"}, 'Location', 'northeast');
    saveas(gcf, filename, 'png');
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

function [img_set, label_set] = extract_img_set(filepath, folder, count)
    
    sz = (256*scale)^2;
    img_set = zeros([sz count]);
    label_set = zeros([1 count]);
    for i=3:count+2
        [img, label] = extract_img(filepath, folder, i);
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