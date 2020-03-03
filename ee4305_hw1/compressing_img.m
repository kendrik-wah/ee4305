n_train = 501;
n_val = 167;

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





[old_dim, old_img, new_dim, new_img] = resize_img(filepath_train, train_folder, 3, 0.5);





function [img, label] = extract_img(filepath, folder, i)
% Extracts the i-th image and its corresponding label as denoted in the given filepath. Only one
% image is extracted at a time.
    filename = filepath + '\\' + folder(i).name;
    img = imread(filename);
    img = img(:);
    tmp = strsplit(filename, {'_', '.'});
    label = str2num(tmp{3});
end

function img = show_img(filepath, folder, i)
    
    filename = filepath + '\\' + folder(i).name;
    img = imread(filename);
    imshow(img, []);
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