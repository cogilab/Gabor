%% Desciprions
% This script is to reproduce figures from main Figure 4. 
% If you want to obtain data, you first need to visit "code_simul" folder.

%% path settings
clear; close; clc;
path0 = '/home/dgxadmin/Minjun/Project-DNN_Gabor';
addpath([path0 '/code_basic'])
basicSettings;

%% Simulation - Original images
%%% parameters
rng(1);
seed_list = 1:20;
domain_name = {'photo', 'art_painting', 'cartoon', 'sketch'};
suffix_list = {'', '_gabor_V1_0.8'};
domain_name = {'photo'};

for dd = 1:length(domain_name)
    disp(domain_name{dd})
    name = domain_name{dd};

    imds = extractImg(pathData_img, ['pacs/' name], 1);
    imds = augmentedImageDatastore([227 227], imds);
    mbq = minibatchqueue(imds,...
            'MiniBatchSize', length(imds.Files),...
            'OutputAsDlarray',[1,0],...
            'MiniBatchFormat',{'SSCB',''},...
            'OutputEnvironment',{'gpu','cpu'});
        
    % accuracy list
    acc_list = zeros(length(suffix_list), length(seed_list));

    reset(mbq);
    [X, Y] = next(mbq);

    % data collection
    for ii = 1:length(suffix_list)
        for jj = 1:length(seed_list)
            disp(jj)
            suffix = suffix_list{ii}; 
            seed = seed_list(jj);
            load([pathResult_simul num2str(seed) 'Trained_nn_' name '_rgb' suffix '.mat'])

            % acc
            YPred = predict(net, X);
            YPred_temp = single(onehotdecode(YPred, 1:net.Layers(24-1, 1).OutputSize, 1)');
            accuracy = sum(YPred_temp == Y) / length(Y);

            acc_list(ii, jj) = accuracy;
        end
    end

    save([pathResult_analysis 'acc_original_' name '.mat'], 'acc_list')
end


%% Simulation - Shape & Texture images
% parameters
rng(1);
seed_list = 1:20;
domain_name = {'photo'};
suffix_list = {'', '_gabor_V1_0.8'};
patch_size = 28;

for dd = 1:length(domain_name)
    disp(domain_name{dd})
    name = domain_name{dd};

    imds = extractImg(pathData_img, ['pacs/' name], 1);
    imds = augmentedImageDatastore([227 227], imds);
    mbq = minibatchqueue(imds,...
            'MiniBatchSize', length(imds.Files),...
            'OutputAsDlarray',[1,0],...
            'MiniBatchFormat',{'SSCB',''},...
            'OutputEnvironment',{'gpu','cpu'});
        
    % accuracy list
    acc_shuffle_list = zeros(length(suffix_list), length(seed_list));
    acc_shape_list = zeros(length(suffix_list), length(seed_list));

    reset(mbq);
    [X, Y] = next(mbq);
    
    X_origin = X;
    
    %%% Shuffled images
    X = extractdata(gather(X));

    parfor tt = 1:size(X, 4)
        img_origin = X(:, :, :, tt);
        patch_num = floor(227 / patch_size);
        img_width_tmp = patch_size * patch_num;
        if patch_num == 1
            img_shuffle=  img_origin;
        else
            img_shuffle = patch_shuffle(img_origin(1:img_width_tmp, 1:img_width_tmp, :), patch_size);
        end
        X(:, :, :, tt) = imresize(img_shuffle, [227 227]);
    end

    X_shuffle = gpuArray(dlarray(X, "SSCB"));

    %%% Shape images
    X = extractdata(gather(X_origin));

    parfor tt = 1:size(X, 4)
        img_origin = X(:, :, :, tt);
        img = fibermetric(rgb2gray(uint8(img_origin)), 5,'ObjectPolarity','dark');
        img(img < 0.7) = 0;
        img = (1 - cat(3, img, img, img)) * 255;
        X(:, :, :, tt) = img;
    end

    X_shape = gpuArray(dlarray(X, "SSCB"));
    
    %%% data collection
    for ii = 1:length(suffix_list)
        for jj = 1:length(seed_list)
            disp(jj)
            suffix = suffix_list{ii}; 
            seed = seed_list(jj);
            load([pathResult_simul num2str(seed) 'Trained_nn_' name '_rgb' suffix '.mat'])

            %%% shuffled acc
            YPred = predict(net, X_shuffle);
            YPred_temp = single(onehotdecode(YPred, 1:net.Layers(24-1, 1).OutputSize, 1)');
            accuracy = sum(YPred_temp == Y) / length(Y);

            acc_shuffle_list(ii, jj) = accuracy;

            %%% shape acc
            YPred = predict(net, X_shape);
            YPred_temp = single(onehotdecode(YPred, 1:net.Layers(24-1, 1).OutputSize, 1)');
            accuracy = sum(YPred_temp == Y) / length(Y);

            acc_shape_list(ii, jj) = accuracy;
        
        end
    end

    save([pathResult_analysis 'acc_shuffled_' name '.mat'], 'acc_shuffle_list')
    save([pathResult_analysis 'acc_shape_' name '.mat'], 'acc_shape_list')
end

%% Fig. 4c, d
suffix_list = {'', '_gabor_V1_0.8'};
suffix_name = {'DNN' ,'GbDNN'};

name_all = {'photo', 'art_painting','cartoon', 'sketch'};
seed_list = 1:20;
d_shape = zeros(length(suffix_list), length(name_all), length(seed_list));
d_shuffle = d_shape;

type = '';

%%% data collection
for dd = 1:length(name_all)
    name = name_all{dd};
    load([pathResult_analysis 'acc_original_' type name '.mat'])
    load([pathResult_analysis 'acc_shape_' type name '.mat'])
    d_shape(:, dd, :) = acc_shape_list - acc_list;
    [h, p] = ttest(d_shape(1, dd, :), d_shape(2, dd, :));
    if mean(d_shape(1, dd, :)) > mean(d_shape(2, dd, :))
        target = 1;
    else
        target = 2;
    end
    fprintf("For %s shape, %s better ,p = %.3f\n", name, suffix_name{target}, p)

    load([pathResult_analysis 'acc_shuffled_' type name '.mat'])
    d_shuffle(:, dd, :) = acc_shuffle_list - acc_list;
    if mean(d_shuffle(1, dd, :)) > mean(d_shuffle(2, dd, :))
        target = 1;
    else
        target = 2;
    end
    [h, p] = ttest(d_shuffle(1, dd, :), d_shuffle(2, dd, :));
    fprintf("For %s texture, %s better ,p = %.3f\n", name, suffix_name{target}, p)
end

%%% Fig. 4c
disp("Shape images")
range = [-80 0];
range_ticks = -80:20:0;
figure;
data = d_shape * 100;
mj_boxplot_dot_line(reshape(data, 2, 80)', palette_network)
data_bar = reshape(data, 2, 80)';
[p, h] = signrank(data_bar(:, 1), data_bar(:, 2))
ylim(range); yticks(range_ticks);
mj_plotctrl([20 30])
exportgraphics(gcf, ['F4/shape_delta_alldomain_acc_summary.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none')

%%% Fig. 4d
range = [-80 -40];
range_ticks = -80:20:-40;
figure;
data = d_shuffle * 100;
mj_boxplot_dot_line(reshape(data, 2, 80)', palette_network)
data_bar = reshape(data, 2, 80)';
[p, h] = signrank(data_bar(:, 1), data_bar(:, 2))
ylim(range); yticks(range_ticks);
mj_plotctrl([20 30])
exportgraphics(gcf, ['F4/shuffle_delta_alldomain_acc_summary.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none')

d_shape = d_shape * 100;
d_shuffle = d_shuffle * 100;

%% Fig. 4e
figure;
marker_list = {'o', '^', 'hexagram', 'pentagram'};
for dd = 1:4
    d_shape_tmp = squeeze(d_shape(:, dd, :)); 
    d_shuffle_tmp = squeeze(d_shuffle(:, dd, :));

    d_shape_tmp = d_shape_tmp - mean(d_shape_tmp(:));
    d_shuffle_tmp = d_shuffle_tmp - mean(d_shuffle_tmp(:));

    d_shape(:, dd, :) = d_shape_tmp;
    d_shuffle(:, dd, :) = d_shuffle_tmp;

    for ii = 1:2
        scatter(squeeze(d_shape(ii, dd, :)), squeeze(d_shuffle(ii, dd, :)), 10, marker_list{dd}, 'filled', ...
            'MarkerFaceColor', palette_network(ii, :))
        hold on
    end
end

data_edge = reshape(d_shape, size(d_shape, 1), size(d_shape, 2) * size(d_shape, 3));
data_corr = reshape(d_shuffle, size(d_shuffle, 1), size(d_shuffle, 2) * size(d_shuffle, 3));

X = [data_corr(1, :)' data_edge(1, :)'; data_corr(2, :)' data_edge(2, :)'];
Y = [ones(1, length(seed_list) * 4) ones(1, length(seed_list) * 4)*2];
mdl = fitcsvm(X, Y, ...
    'KernelFunction', 'linear', 'BoxConstraint', 10);

beta = mdl.Beta;
b = mdl.Bias;

xlim([-15 25]); xticks(-15:10:25)
ylim([-10 15]); yticks(-10:10:15);
X1 = linspace(-15, 25 ,100);
X2 = -(beta(1)/beta(2)*X1)-b/beta(2);
plot(X1,X2,'--', 'Color', [.3 .3 .3], 'LineWidth', 2)
mj_plotctrl([30 30])
exportgraphics(gcf, ['F4/shape_texture_bias.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none')


%% subfunctions
function shuffled_image = patch_shuffle(image, patch_size)
    % Get the size of the image
    [height, width, channels] = size(image);
    
    % Calculate the number of patches along each dimension
    num_patches_row = ceil(height / patch_size);
    num_patches_col = ceil(width / patch_size);
    
    % Initialize the shuffled image
    shuffled_image = zeros(size(image), 'like', image);
    
    % Create an array of patch indices and shuffle it
    thr_list = [0 4.2426   13.6382   31.2090   59.3801   99.5188  157.0796  229.4690];
    thr = thr_list(num_patches_col); d = 0; origin = 1:(num_patches_col * num_patches_row);
    while d < thr
        shuffled_indices = randperm(num_patches_row * num_patches_col);
        while sum(abs(shuffled_indices- (1:(num_patches_col * num_patches_row))), 'all') == 0
            shuffled_indices = randperm(num_patches_row * num_patches_col);
        end
        d = sqrt(sum((origin - shuffled_indices).^2));
    end
    
    if patch_size == 1
        shuffled_image = image(randperm(height), randperm(width), :);
    else
        % Loop through each patch and place it in the shuffled position
        it = 1;
        for i = 1:num_patches_row
            for j = 1:num_patches_col
                % Calculate the coordinates of the current patch
                row_start = (i-1) * patch_size + 1;
                row_end = min(i * patch_size, height);
                col_start = (j-1) * patch_size + 1;
                col_end = min(j * patch_size, width);
                
                patch = image(row_start:row_end, col_start:col_end, :);
                
    
                % Calculate the coordinates of the current patch
                [new_i, new_j] = ind2sub([num_patches_row num_patches_col], shuffled_indices(it)); it = it + 1;
                row_start = (new_i - 1) * patch_size + 1;
                row_end = min(new_i * patch_size, height);
                col_start = (new_j-1) * patch_size + 1;
                col_end = min(new_j * patch_size, width);
    
                shuffled_image(row_start:row_end, col_start:col_end, :) = patch;
            end
        end
    end
end

function indices = divide_indices(length, num_divisions)
    % Calculate the size of each division
    division_size = floor(length / num_divisions);
    
    % Generate the indices
    indices = 1:(division_size-1):length;
    
    % Ensure the last index is the length itself
    if indices(end) ~= length
        indices = [indices, length];
    end
end