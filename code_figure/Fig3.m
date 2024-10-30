%% Desciprions
% This script is to reproduce figures from main Figure 3.
% If you want to obtain data, you first need to visit "code_simul" folder.

%% path settings
clear; close; clc;
path0 = '/home/dgxadmin/Minjun/Project-DNN_Gabor';
addpath([path0 '/code_basic'])
basicSettings;

%% Fig. 3a - Example clustering
rng(1) 
close all;

name_all = {'photo', 'art_painting', 'cartoon', 'sketch'};
suffix_list = {'', '_gabor_V1_0.8'};
numChannel = 3;

is_display = 1;
domain_ids = [1 4];
name = name_all(domain_ids);

seed_select = 4; 
class_select = [2 6];
layer_select = 18;

palette_tmp = palette_domain([1 3], :);
target = [211         959        2021        3240];

dist_imgs = cell(1, length(suffix_list));

imd_list = cell(1, length(name));
label_list = cell(length(name), 1);
file_list = [];
for jj = 1:length(name)
    portion_list = [1 1670/2048 1670/2344 1670/3928];
    portion = portion_list(domain_ids(jj));

    imds = extractImg(pathData_img, ['pacs/' name{jj}], portion);
    if numChannel == 3; option = 'gray2rgb';
    elseif numChannel == 1; option = 'rgb2gray'; end
    augimds = augmentedImageDatastore([227 227], imds, 'ColorPreprocessing',option);
    mbq = minibatchqueue(augimds,...
        'MiniBatchSize', length(augimds.Files),...
        'OutputAsDlarray',[1,0],...
        'MiniBatchFormat',{'SSCB',''},...
        'OutputEnvironment',{'gpu','cpu'});

    [X, ~] = next(mbq);
    imd_list{jj} = X;
    label_list{jj} = imds.Labels;
    file_list = [file_list; imds.Files];
end 

for ii = 1:length(suffix_list)
    suffix = suffix_list{ii};
    
    load([pathResult_simul num2str(seed_select) 'Trained_nn_' name{1} '_' name{2} '_rgb' suffix '.mat']);
    subnet = dlnetwork(net.Layers(1:layer_select));
    
    act_list = cell(length(name), 1);

    
    for jj = 1:length(name)
        X = imd_list{jj};
        act_list{jj} = gather(extractdata(predict(subnet, X)));
    end

    act1 = act_list{1}; 
    act2 = act_list{2};

    act_con = [act1'; act2'];
    act_con_latent = tsne(act_con);


    dist_imgs{ii} = pdist2(act_con_latent, act_con_latent);

    label_domain = [ones(1, length(label_list{1})) 2*ones(1, length(label_list{2}))]';
    label_class = [label_list{1}; label_list{2}];

    ind_select = find(label_class == class_select(1) | label_class == class_select(2));
    sil_domain = calcuSilhouetteIndex(act_con_latent(ind_select, :), label_domain(ind_select));
    sil_class = calcuSilhouetteIndex(act_con_latent(ind_select, :), label_class(ind_select));

    disp(ii)
    disp(sil_domain)
    disp(sil_class)

    f = figure; f.Position = [300 500 330 350];
    for cc = 1:length(class_select)
        act_tmp = act_con_latent(1:length(label_list{1}), :);
        act_tmp = act_tmp(label_list{1} == class_select(cc), :);

        s = scatter(act_tmp(:, 1), act_tmp(:, 2), 20, palette_tmp(cc, :), '+');
        s.MarkerFaceAlpha = 0.5;
        s.MarkerEdgeAlpha = 0.5;
        hold on

        if cc==1; c_tmp = [.1 .1 .1]; else c_tmp = [0 1 0]; end
        s = scatter(act_con_latent(target(cc), 1), act_con_latent(target(cc), 2), 20, c_tmp, '+');

    end
    for cc = 1:length(class_select)
        act_tmp = act_con_latent(length(label_list{1})+1:end, :);
        act_tmp = act_tmp(label_list{2} == class_select(cc), :);

        s = scatter(act_tmp(:, 1), act_tmp(:, 2), 5, palette_tmp(cc, :), 'o', 'filled');
        s.MarkerFaceAlpha = 0.8;
        s.MarkerEdgeAlpha = 0.8;

        if cc==1; c_tmp = [.1 .1 .1]; else c_tmp = [0 1 0]; end
        s = scatter(act_con_latent(target(cc+2), 1), act_con_latent(target(cc+2), 2), 5, c_tmp, 'o', 'filled');

    end

    set(gca,'TickDir','out');
    box off; 
    ylabel('t-SNE axis 2')
end
toc


figure(1)
range = [-0.8 0.8] * 1e10;
xticks(range); xlim(range)
yticks(range); ylim(range)
mj_plotctrl([40 40])
exportgraphics(gcf, 'F3/PS_clustering_DNN.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none')
figure(2)
range = [-1.5 1.5] * 1e10;
xticks(range); xlim(range)
yticks(range); ylim(range)
mj_plotctrl([40 40])
exportgraphics(gcf, 'F3/PS_clustering_GbDNN.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none')
figure(3)
for ii = 1:length(target)
    subplot(2, length(target)/2, ii)
    img = imread(file_list{target(ii)});
    imagesc(rescale(img))
    axis off; axis equal
end
exportgraphics(gcf, 'F3/sample_images.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none')


%% Simulation - calculating SI
name_all = {'photo', 'art_painting', 'cartoon', 'sketch'};

layer = 18; % fc6
pairs = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4; 4 3; 4 2; 4 1; 3 2; 3 1; 2 1];

seed_list = 1:20;
numChannel = 3;

sil_tot_domain = zeros(length(suffix_list), length(pairs), length(seed_list), 1, 2);
sil_tot_class = zeros(length(suffix_list), length(pairs), length(seed_list), 1, 2);

for pp = 1:length(pairs)
    name = {name_all{pairs(pp, 1)}, name_all{pairs(pp, 2)}};

    for ii = 1:length(suffix_list)
        suffix = suffix_list{ii};
        fprintf('suffix = %s\n', suffix)
        
        sil_domain_tmp = zeros(length(seed_list), 2);
        sil_class_tmp = zeros(length(seed_list), 2);
        parfor ss = 1:length(seed_list)
            seed= seed_list(ss);
            fprintf('seed = %d\n', seed)
            [sil_domain, sil_class] = runCalcSilhouette(pathData_img, pathResult_simul, seed, suffix, layer, name, numChannel);
            sil_domain_tmp(ss, :) = sil_domain;
            sil_class_tmp(ss, :) = sil_class;
        end
        sil_tot_domain(ii, pp, :, 1, :) = sil_domain_tmp;
        sil_tot_class(ii, pp, :, 1, :) = sil_class_tmp;
    end
end

save([pathResult_analysis 'Sil_tot_domain_allpairs_uncontrol.mat'], 'sil_tot_domain')
save([pathResult_analysis 'Sil_tot_class_allpairs_uncontrol.mat'], 'sil_tot_class')


%% Fig. 3b-c
%%% Fig. 3b
figure;
load([pathResult_analysis 'Sil_tot_class_allpairs_uncontrol.mat'])
data_all = squeeze(sil_tot_class(:, :, :, 1, 2));

data_bar = squeeze(data_all(:, 3, :));
x = 1:2:2*length(suffix_list);
mj_boxplot(data_bar', palette_network);
hold on

set(gca, 'tickdir', 'out')
ylim([-0.06 0.15])
yticks([-0.05:0.05:0.15])
ylabel('silhouette index')

mj_plotctrl([20 40])
exportgraphics(gcf, 'F3/PS_SI_boxplot_class.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none')
disp(data_all(:, 3, 4))

%%% Fig. 3c
figure;
load([pathResult_analysis 'Sil_tot_domain_allpairs_uncontrol.mat'])
data_all = squeeze(sil_tot_domain(:, :, :, 1, 2));

data_bar = squeeze(data_all(:, 3, :));
x = 1:2:2*length(suffix_list);
mj_boxplot(data_bar', palette_network);
hold on

set(gca, 'tickdir', 'out')
ylim([-0.05 0.35])
yticks([-0.05:0.1:0.35])
ylabel('silhouette index')

mj_plotctrl([20 40])
exportgraphics(gcf, 'F3/PS_SI_boxplot_domain.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none')
disp(data_all(:, 3, 4))


%% Fig. 3d-e
%%% Fig. 3d
figure;
load([pathResult_analysis 'Sil_tot_class_allpairs_uncontrol.mat'])
data_all = squeeze(sil_tot_class(:, :, :, 1, 2));
data_all_srt = data_all(:, ind_ref, :); data_all_srt = data_all_srt(:, 1:6, :);

mj_boxplot_dot_line_verMean(data_all_srt, palette_network)
ylim([-0.2 0.2]); yticks(-0.2:0.1:0.2); 
ylabel('accuracy (%)'); 
box off
mj_plotctrl([20 40])
exportgraphics(gcf, ['F3/Alldomain_SI_class.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none')
[h, p] = signrank(reshape(data_all_srt(1, :, :), 1, []), reshape(data_all_srt(2, :, :), 1, []))

%%% Fig. 3e
figure;
load([pathResult_analysis 'Sil_tot_domain_allpairs_uncontrol.mat'])
data_all = squeeze(sil_tot_domain(:, :, :, 1, 2));
data_all_srt = data_all(:, ind_ref, :); data_all_srt = data_all_srt(:, 1:6, :);

mj_boxplot_dot_line_verMean(data_all_srt, palette_network)
ylim([-0.1 0.5]); yticks(-0.1:0.1:0.5); 
ylabel('accuracy (%)'); 
box off
mj_plotctrl([20 40])
exportgraphics(gcf, ['F3/Alldomain_SI_domain.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none')
[h, p] = signrank(reshape(data_all_srt(1, :, :), 1, []), reshape(data_all_srt(2, :, :), 1, []))

%% Fig. 3f-g
%%% Data collection - accuracy diff.
domain_name = {'photo', 'art_painting', 'cartoon', 'sketch'};
pairs = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4;4 3; 4 2; 4 1; 3 2; 3 1; 2 1];

suffix_list = {'', '_gabor_V1_0.8'};
suffix_name = {'DNN', 'GbDNN'};

seed_list = 1:20;
data_all = zeros(length(suffix_list), size(pairs, 1), length(seed_list));

for ii = 1:length(suffix_list)
    for pp = 1:size(pairs, 1)
        for jj = 1:length(seed_list)
            suffix = suffix_list{ii};
            seed = seed_list(jj);

            name1 = domain_name{pairs(pp, 1)};
            name2 = domain_name{pairs(pp, 2)};
            
            result1 = load([pathResult_simul num2str(seed) 'Trained_result_' name1 '_rgb' suffix '.mat']);
            result2 = load([pathResult_simul num2str(seed) 'Trained_result_' name1 '_' name2 '_rgb' suffix '.mat']);

            data_all(ii, pp, jj) = result2.result.ValidationAccuracy2(end, 2);
        end
    end
end
data_diff = squeeze(data_all(2, :, :) - data_all(1, : ,:));
[~, ind_ref] = sort(mean(data_diff, 2), 'descend');
finalacc_diff_all = squeeze(data_all(2, ind_ref, :) - data_all(1, ind_ref, :));

%%% Fig. 3f
f = figure; 
color = [.3 .3 .3];

data_all = squeeze(sil_tot_class(:, :, :, 1, 2));
data_all_srt = data_all(:, ind_ref, :); data_all_srt = data_all_srt(:, :, :);
silhouette_diff_all = squeeze(data_all_srt(2, :, :) - data_all_srt(1, :, :));

select = 1:12;

y = squeeze(mean(finalacc_diff_all(select, :), 2));
x = squeeze(mean(silhouette_diff_all(select, :), 2));

y_err = std(squeeze(finalacc_diff_all(select, :)), [], 2);
x_err = std(squeeze(silhouette_diff_all(select, :)), [], 2);

plot(x, y, '.', 'Color', color, 'MarkerSize', 12); hold on
errorbar(x, y, x_err', 'horizontal', 'CapSize', 0, 'LineWidth', 0.5', 'Color', color, 'LineStyle','none')
errorbar(x, y, y_err', 'vertical', 'CapSize', 0, 'LineWidth', 0.5', 'Color', color, 'LineStyle','none')

[r, p] = corr(x, y)

mdl = fitlm(x, y);
intercept = mdl.Coefficients(1, 1).Estimate; b = mdl.Coefficients(2, 1).Estimate;
x = -0.1:0.01:0.2; y = x * b + intercept;

plot(x, y, '--', 'Color', color, 'LineWidth', 1)
ylim([-10 40]); yticks([-10:10:40]);
xlim([-0.1 0.2]); xticks(-0.1:0.1:0.2);

mj_plotctrl([40 40])
exportgraphics(gcf, 'F3/correlation_class.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none')

%%% Fig. 3g
f = figure; 
color = [.3 .3 .3];

data_all = squeeze(sil_tot_domain(:, :, :, 1, 2));
data_all_srt = data_all(:, ind_ref, :); data_all_srt = data_all_srt(:, :, :);
silhouette_diff_all = squeeze(data_all_srt(2, :, :) - data_all_srt(1, :, :));

select = 1:12;

y = squeeze(mean(finalacc_diff_all(select, :), 2));
x = squeeze(mean(silhouette_diff_all(select, :), 2));

y_err = std(squeeze(finalacc_diff_all(select, :)), [], 2);
x_err = std(squeeze(silhouette_diff_all(select, :)), [], 2);

plot(x, y, '.', 'Color', color, 'MarkerSize', 12); hold on
errorbar(x, y, x_err', 'horizontal', 'CapSize', 0, 'LineWidth', 0.5', 'Color', color, 'LineStyle','none')
errorbar(x, y, y_err', 'vertical', 'CapSize', 0, 'LineWidth', 0.5', 'Color', color, 'LineStyle','none')

[r, p] = corr(x, y)

mdl = fitlm(x, y);
intercept = mdl.Coefficients(1, 1).Estimate; b = mdl.Coefficients(2, 1).Estimate;
x = -0.3:0.01:0.3; y = x * b + intercept;

plot(x, y, '--', 'Color', color, 'LineWidth', 1)
ylim([-10 40]); yticks([-10:10:40]);
xlim([-0.3 0.2]); xticks(-0.3:0.1:0.2);

mj_plotctrl([40 40])
exportgraphics(gcf, 'F3/correlation_domain.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none')

%% subfunction
function [sil_domain, sil_class] = runCalcSilhouette(pathData_img, pathResult_simul, seed, suffix, layer, name, numChannel)
    % outputs
    sil_domain = zeros(1, 2); sil_class = zeros(1, 2);

    % load networks
    net1 = load([pathResult_simul num2str(seed) 'Trained_nn_' name{1} '_rgb' suffix '.mat']);
    net2 = load([pathResult_simul num2str(seed) 'Trained_nn_' name{1} '_' name{2} '_rgb' suffix '.mat']);
    net1 = net1.net; net2 = net2.net;
    
    act_list = cell(length(name), 2);
    label_list = cell(length(name), 2);

    for jj = 1:length(name)
        % load images
        switch name{jj}
            case 'photo'
                portion = 1;
            case 'art_painting'
                portion = 1670/2048;
            case 'cartoon'
                portion = 1670/2344;
            case 'sketch'
                portion = 1670/3928;
        end
        portion = 1;

        imds = extractImg(pathData_img, ['pacs/' name{jj}], portion);

        if numChannel == 3; option = 'gray2rgb';
        elseif numChannel == 1; option = 'rgb2gray'; end

        augimds = augmentedImageDatastore([227 227], imds, 'ColorPreprocessing',option);
        
        mbq = minibatchqueue(augimds,...
            'MiniBatchSize', length(augimds.Files),...
            'OutputAsDlarray',[1,0],...
            'MiniBatchFormat',{'SSCB',''},...
            'OutputEnvironment',{'gpu','cpu'});

        subnet1 = dlnetwork(net1.Layers(1:layer));
        subnet2 = dlnetwork(net2.Layers(1:layer));
    
        reset(mbq);
        [X, ~] = next(mbq);

        act1_tot = gather(extractdata(predict(subnet1, X)));
        act2_tot = gather(extractdata(predict(subnet2, X)));

        act_list{jj, 1} = act1_tot; 
        act_list{jj, 2} = act2_tot;
        label_list{jj} = imds.Labels;
        label_list{jj} = imds.Labels;
    end 
    
    %%% main silhouette index calculation
    for tt = 1:2
        act1 = act_list{1, tt}; 
        act2 = act_list{2, tt};
    
        act_con = [act1'; act2'];
        act_con_pca = tsne(act_con);

        label_domain = [ones(1, length(label_list{1})) 2*ones(1, length(label_list{2}))]';
        label_class = [label_list{1}; label_list{2}];
    
        sil_domain(tt) = calcuSilhouetteIndex(act_con_pca, label_domain);
        sil_class(tt) = calcuSilhouetteIndex(act_con_pca, label_class);
    end
end

function [sil_domain, sil_class] = runCalcDBI(pathData_img, pathResult_simul, seed, suffix, layer, name, numChannel)
    % outputs
    sil_domain = zeros(1, 2); sil_class = zeros(1, 2);

    % load networks
    net1 = load([pathResult_simul num2str(seed) 'Trained_nn_' name{1} '_rgb' suffix '.mat']);
    net2 = load([pathResult_simul num2str(seed) 'Trained_nn_' name{1} '_' name{2} '_rgb' suffix '.mat']);
    net1 = net1.net; net2 = net2.net;
    
    act_list = cell(length(name), 2);
    label_list = cell(length(name), 2);

    for jj = 1:length(name)
        % load images
        switch name{jj}
            case 'photo'
                portion = 1;
            case 'art_painting'
                portion = 1670/2048;
            case 'cartoon'
                portion = 1670/2344;
            case 'sketch'
                portion = 1670/3928;
        end
        portion = 1;

        imds = extractImg(pathData_img, ['pacs/' name{jj}], portion);

        if numChannel == 3; option = 'gray2rgb';
        elseif numChannel == 1; option = 'rgb2gray'; end

        augimds = augmentedImageDatastore([227 227], imds, 'ColorPreprocessing',option);
        
        mbq = minibatchqueue(augimds,...
            'MiniBatchSize', length(augimds.Files),...
            'OutputAsDlarray',[1,0],...
            'MiniBatchFormat',{'SSCB',''},...
            'OutputEnvironment',{'gpu','cpu'});

        subnet1 = dlnetwork(net1.Layers(1:layer));
        subnet2 = dlnetwork(net2.Layers(1:layer));
    
        reset(mbq);
        [X, ~] = next(mbq);

        act1_tot = gather(extractdata(predict(subnet1, X)));
        act2_tot = gather(extractdata(predict(subnet2, X)));

        act_list{jj, 1} = act1_tot; 
        act_list{jj, 2} = act2_tot;
        label_list{jj} = imds.Labels;
        label_list{jj} = imds.Labels;
    end 
    
    %%% main silhouette index calculation
    for tt = 1:2
        act1 = act_list{1, tt}; 
        act2 = act_list{2, tt};
    
        act_con = [act1'; act2'];
        act_con_pca = tsne(act_con);

        label_domain = [ones(1, length(label_list{1})) 2*ones(1, length(label_list{2}))]';
        label_class = [label_list{1}; label_list{2}];
    
        sil_domain(tt) = calcuDBIndex(act_con_pca, label_domain);
        sil_class(tt) = calcuDBIndex(act_con_pca, label_class);
    end
end

