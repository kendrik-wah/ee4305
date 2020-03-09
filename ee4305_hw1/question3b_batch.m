clear;

% train has 501 items. val has 167 items.
n_train = 501;
n_val = 167;
eta = 0.005;
epochs = 100;
epoch_test = [0:20:200];
val_acc = zeros([1 length(epoch_test)]);
scales = [1, 0.75, 0.5, 0.25];
training_sets = [1:1:501];

% Defining training data
filepath_train = "group_3\\train";
train_folder = dir(filepath_train);

% Retrieve validation data
filepath_val = "group_3\\val";
val_folder = dir(filepath_val);




for h=1:length(scales)
    
    dim = 256*scales(h);

   % redefining the data sets according to scales, full clarity
   [training_data, training_label] = extract_img_set(filepath_train, train_folder, n_train, scales(h));
   [validation_data, validation_label] = extract_img_set(filepath_val, val_folder, n_val, scales(h));
   epoch = [1:1:epochs];
   
   n_components = extract_component_numbers(filepath_train, train_folder, n_train, dim);
   
   dimensions_to_test = [1, ceil(n_components/20), ceil(n_components/10), ceil(n_components/5), ceil(n_components/2), n_components, min(dim-1, n_components+20)];
   display(dimensions_to_test);
   
   for i=1:length(dimensions_to_test)
       
       display("dimensions:", num2str(dimensions_to_test(i)));
       
       train_pca_data = zeros([dim^2, n_train]);
       accu_train = zeros(1, length(epoch_test));
       accu_val = zeros(1, length(epoch_test));
       
       for l=3:n_train+2
           [coeff, img, x_form] = pca_img(filepath_train, train_folder, l, dimensions_to_test(i), scales(h));
           train_pca_data(:,l-2) = x_form(:);
       end
       
       for j=1:length(epoch_test)

           net = perceptron;
           epochs = epoch_test(j);
           net.trainParam.epochs=epochs;
           epoch = [0:1:epochs-1];

           net = train(net, train_pca_data, training_label);

           train_count = [1:1:n_train];
           train_out = net(training_data);

           val_count = [1:1:n_val];
           val_out = net(validation_data);

           accuracy_train = 1 - mean(abs(train_out - training_label));
           accuracy_val = 1 - mean(abs(val_out - validation_label)); 
           
           accu_train(j) = accuracy_train;
           accu_val(j) = accuracy_val;
       end
       
       display("Components: ", num2str(dimensions_to_test(i)));

       display("training accuracies: ");
       display(accu_train);

       display("validation accuracies: ");
       display(accu_val);
       
       filename = sprintf("q3b_batch\\batch_accuracy_%dpx_%d_comps",dim, dimensions_to_test(i));
       plot(epoch_test, accu_train, epoch_test, accu_val);
       xlabel("epochs");
       ylabel("accuracy");
       legend({"training", "validation"}, 'Location', 'northeast');
       saveas(gcf, filename, 'png');
   end
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

function [coeff, img, x_form] = pca_img(filepath, folder, i, nComp, scale)
    filename = filepath + '\\' + folder(i).name;
    old_img = imread(filename);
    img = double(old_img);
    img = imresize(img, scale);
    img_mean = mean(img);
    img_adjusted = img-img_mean;
    
    [coeff, score] = pca(img_adjusted);
    x_form = score(:,1:nComp)*coeff(:,1:nComp)';
    x_form = x_form + img_mean;
    x_form = uint8(x_form);
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

% Possible improvements that can be made:
% 1) Change performFcn and see what happens.
% 2) Change number of epochs to below 66 and then see the overall error.