%% Desciprions
% This script is to reproduce figures from supplementary Figure 4.
% If you want to obtain data, you first need to visit "code_simul" folder.

%% path settings
clear; close; clc;
path0 = '/home/dgxadmin/Minjun/Project-DNN_Gabor';
addpath([path0 '/code_basic'])
basicSettings;

%% Simulation - dimensionality measure
%%% parameters
rng(1) 

seed_list = 1:20;
suffix_list = {'', '_gabor_V1_0.8'};
suffix_name = {'DNN', 'DNN+Gabor'};

numChannel = 3;
layer_list = [3 7 11 13 15 18 21];
name_all = {'photo', 'art_painting', 'cartoon', 'sketch'};

dims_tot = zeros(length(name_all), length(layer_list), length(suffix_list), length(seed_list));
svs_tot = cell(length(name_all));

%%% data collection
for dd = 1:length(name_all)
    name = name_all{dd};
    disp(name)
    
    % load images
    portion_list = [1 1670/2048 1670/2344 1670/3928];
    portion = portion_list(dd);

    imds = extractImg(pathData_img, ['pacs/' name], portion);
    imds.Labels = categorical(imds.Labels);

    if numChannel == 3; option = 'gray2rgb';
    elseif numChannel == 1; option = 'rgb2gray'; end

    augimds = augmentedImageDatastore([227 227], imds, 'ColorPreprocessing',option);           
    mbq = minibatchqueue(augimds,...
        'MiniBatchSize', length(augimds.Files),...
        'OutputAsDlarray',[1,0],...
        'MiniBatchFormat',{'SSCB',''},...
        'OutputEnvironment',{'gpu','cpu'});
    
    shuffle(mbq);
    [X, ~] = next(mbq);
    
    singular_vals = zeros(length(layer_list), length(suffix_list), length(seed_list), size(X, 4));
    
    for ss = 1:length(seed_list)
        seed= seed_list(ss);
        fprintf('seed = %d\n', seed)
        
        for ii = 1:length(suffix_list)
            suffix = suffix_list{ii};
            fprintf('suffix = %s\n', suffix)
            
            % load networks
            load([pathResult_simul num2str(seed) 'Trained_nn_' name '_rgb' suffix '.mat']);

            for ll = 1:length(layer_list)
                layer = layer_list(ll);
                subnet = dlnetwork(net.Layers(1:layer));
    
                % reshape to 2D
                act = predict(subnet, X);
                act = extractdata(gather(act));
                act_size = size(act);
                act_reshape = reshape(act, prod(act_size(1:end-1)), act_size(end));
                
                [~,~,latent] = pca(act_reshape);
    
                singular_vals(ll, ii, ss, :) = latent;
                dims_tot(dd, ll, ii, ss) = sum(latent)^2 / sum(latent .^2);
            end
        end
    end
    svs_tot{dd} = singular_vals;
end

save([pathResult_analysis 'dims_tot'], 'dims_tot')
save([pathResult_analysis 'SVs_tot'], 'svs_tot')


%% Fig S4c
seed_list = 1:20;
suffix_list = {'', '_gabor_V1_0.8'};
suffix_name = {'DNN', 'DNN+Gabor'};

numChannel = 3;
layer_list = [3 7 11 13 15 18 21];
name_all = {'photo', 'art_painting', 'cartoon', 'sketch'};

load([pathResult_analysis 'dims_tot'])
load([pathResult_analysis 'SVs_tot'])

ll = 1;

%%% Spectra and Dimensionality
for dd = 1:length(name_all)
    figure;
    svs = svs_tot{dd}; svs = squeeze(svs(ll, :, :, 1:1000));
    for ii = 1:length(suffix_list) 
        svs_data = squeeze(svs(ii, :, :))';
        svs_data = svs_data ./ repmat(svs_data(1, :), 1000, 1);
        mj_shadedplot(1:1000, svs_data, palette_network(ii, :)); hold on
    end
    xlabel('order'); ylabel('eigenvalues');
    xticks([1 10 100 1000]); xlim([1 1000])
    yticks([1e-6 1e-4 1e-2 1e0]); ylim([1e-6 1])
    mj_plotctrl([18 22]); 
    set(gca,'xscale', 'log', 'yscale', 'log')
    exportgraphics(gcf, ['F3/Spectra_' name_all{dd} '.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none')
    
    figure;
    x = 1:2:2*length(suffix_list);
    data_bar = squeeze(dims_tot(dd, 1, :, :));
    mj_boxplot(data_bar', palette_network);
    hold on; box off
    ylim([0 8]); yticks([0 8])

    if dd == 1
        ylim([1 4]); yticks([1 4])
    elseif dd == 4
        ylim([0 6]); yticks([0 6])
    end
    mj_plotctrl([5 8]); 
    ylabel('ED')
    exportgraphics(gcf, ['F3/ED_' name_all{dd} '.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none')
end


%% Fig. S4d
%%% spectra
load([pathResult_analysis 'dims_tot'])
load([pathResult_analysis 'SVs_tot'])

figure; hold on
ll = 1;
for ii = 1:2
    svs_net = [];
    for dd = 1:length(name_all)
        svs = svs_tot{dd}; svs = squeeze(svs(ll, :, :, 1:1000));
        svs_net = [svs_net squeeze(svs(ii, :, :))'];
    end
   
    svs_net = svs_net ./ repmat(svs_net(1, :), 1000, 1);
    x = 1:1000;
    y_mean = mean(svs_net');
    y_err = std(svs_net') * 0.99;

    plot(x, y_mean, 'Color', palette_network(ii, :), 'LineWidth', 1.5);  

    patch([x flip(x)], [y_mean-y_err flip(y_mean+y_err)], palette_network(ii, :), ...
    'FaceAlpha',0.3, 'EdgeColor','none'); 

    set(gca,'xscale', 'log', 'yscale', 'log')
    xlabel('order'); ylabel('eigenvalues');
    xticks([1 10 100 1000]); xlim([1 1000])
    yticks([1e-6 1e-4 1e-2 1e0]); ylim([1e-6 1])
    mj_plotctrl([30 30]); 
    set(gca,'xscale', 'log', 'yscale', 'log')
end
exportgraphics(gcf, ['F3/Spectra_summary.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none')

%%% Dimensionality
figure;
dims_all = reshape(permute(dims_tot(:, 1, :, :), [3 1 4 2]), length(suffix_list), []);
mj_boxplot(dims_all', palette_network);
mj_plotctrl([15 30])
ylim([0 8]); yticks([0 4 8])
exportgraphics(gcf, ['F3/ED_summary.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none')
