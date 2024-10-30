%% Desciprions
% This script is to reproduce figures from main Figure 2.
% If you want to obtain data, you first need to visit "code_simul" folder.

%% path settings
clear; close; clc;
path0 = '/home/dgxadmin/Minjun/Project-DNN_Gabor';
addpath([path0 '/code_basic'])
basicSettings;

%% Fig. 2c - Example learning curve
% Parameters
domain_ids = [1 4]; % photo and sketch
domain_name = {'photo', 'art_painting', 'cartoon', 'sketch'};
name1 = domain_name{domain_ids(1)};
name2 = domain_name{domain_ids(2)};

seed_list = 1:20;
seed_select = 1;

control = 0;
include_second = 1;

suffix_list = {'', '_gabor_V1_0.8'};
numEpoch = 90;

%%% data collection
data_all = zeros(length(suffix_list), numEpoch*2, length(seed_list));
data_second = zeros(length(suffix_list), numEpoch, length(seed_list));

for ii = 1:length(suffix_list)
    for jj = 1:length(seed_list)
        data_tmp = [];

        suffix = suffix_list{ii};
        seed = seed_list(jj);

        result1 = load([pathResult_simul num2str(seed) 'Trained_result_' name1 '_rgb' suffix '.mat']);
        result2 = load([pathResult_simul num2str(seed) 'Trained_result_' name1 '_' name2 '_rgb' suffix '.mat']);

        len = length(result1.result.ValidationAccuracy(:, 2));
        step = floor(len/numEpoch);
        data_tmp = result1.result.ValidationAccuracy(1:step:step*numEpoch-1, 2)';

        len = length(result2.result.ValidationAccuracy2(:, 2));
        step = floor(len/numEpoch);
        data_tmp = [data_tmp result2.result.ValidationAccuracy2(1:step:step*numEpoch-1, 2)'];
        
        data_all(ii, :, jj) = data_tmp;
        data_second(ii, :, jj) = result2.result.ValidationAccuracy1(1:step:step*numEpoch-1, 2)';
    end
end

%%% Plotting - top panel
f = figure; f.Position = [300 500 267 178]; hold on
for ii = 1:length(suffix_list)

    data = squeeze(data_all(ii, :, :));
    data_second_tmp = squeeze(data_second(ii, :, :));

    if length(seed_select) == 1
        plot(1:numEpoch*2, data(:, seed_select), 'Color', palette_network(ii, :), 'LineWidth', 2)
    else
        mj_shadedplot(1:numEpoch*2, data(:, seed_select),  palette_network(ii, :), 2)
    end
end

% chance level
plot(1:numEpoch*2, ones(1, numEpoch*2) * 16.1, '--', 'Color', [.33 .33 .33], 'LineWidth', 2)

% domain patches
x1 = 1:numEpoch;
patch([x1 flip(x1)], [zeros(1, length(x1)) ones(1, length(x1)) * 100], palette_domain(domain_ids(1), :), ...
       'FaceAlpha',0.05, 'EdgeColor','none')
x2 = numEpoch:numEpoch*2; hold on
patch([x2 flip(x2)], [zeros(1, length(x2)) ones(1, length(x2)) * 100], palette_domain(domain_ids(2), :), ...
       'FaceAlpha',0.05, 'EdgeColor','none')

xlim([1 numEpoch*2])
xticks([1 numEpoch/2 numEpoch numEpoch*1.5 numEpoch*2])
ylim([0 100])
yticks([0 50 100])
xlabel('epoch')
ylabel('photo accuracy (%)')

mj_plotctrl([50 14])
exportgraphics(gcf, 'F2/PS_learningcurve_first.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none')

%%% Plotting - bottom panel
f = figure; f.Position = [300 500 267 178]; hold on
for ii = 1:length(suffix_list)

    data = squeeze(data_all(ii, :, :));
    data_second_tmp = squeeze(data_second(ii, :, :));
    plot(numEpoch+1:numEpoch*2, data_second_tmp(:, seed_select), '--', 'Color', palette_network(ii, :), 'LineWidth', 2)
end

% chance level
plot(1:numEpoch*2, ones(1, numEpoch*2) * 16.1, '--', 'Color', [.33 .33 .33], 'LineWidth', 2)

% domain patches
x1 = 1:numEpoch;
patch([x1 flip(x1)], [zeros(1, length(x1)) ones(1, length(x1)) * 100], palette_domain(domain_ids(1), :), ...
       'FaceAlpha',0.05, 'EdgeColor','none')
x2 = numEpoch:numEpoch*2; hold on
patch([x2 flip(x2)], [zeros(1, length(x2)) ones(1, length(x2)) * 100], palette_domain(domain_ids(2), :), ...
       'FaceAlpha',0.05, 'EdgeColor','none')

xlim([1 numEpoch*2])
xticks([1 numEpoch/2 numEpoch numEpoch*1.5 numEpoch*2])
ylim([0 100])
yticks([0 50 100])
xlabel('epoch')
ylabel('photo accuracy (%)')

mj_plotctrl([50 14])
exportgraphics(gcf, 'F2/PS_learningcurve_second.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none')

%% Fig. 2d - box plot
figure;

data_bar = squeeze(data_all(:, end, :));
x = 1:2:2*length(suffix_list);
mj_boxplot(data_bar', palette_network);
hold on

set(gca, 'tickdir', 'out'); box off
ylim([0 60])
yticks(0:20:60)
ylabel('final accuracy (%)')
yline(16.1, '--', 'Color', [.33 .33 .33], 'LineWidth', 2)
mj_plotctrl([15 30])
exportgraphics(gcf, 'F2/PS_boxplot.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none')

%% Fig. 2e - Filter controls
%%% paramters
domain_ids = [1 4]; % photo and sketch
domain_name = {'photo', 'art_painting', 'cartoon', 'sketch'};
name1 = domain_name{domain_ids(1)};
name2 = domain_name{domain_ids(2)};

seed_list = 1:20;

suffix_list = {'_randfix', '_gabor_V1_0.8_pxshuffle', '_gabor_V1_0.8_unfix', '_gabor_V1_0.8'};
palette_network_tmp = [palette_network(1, :); palette_network(1, :); palette_network(1, :); palette_network(2, :)];

numEpoch = 90;


%%% data collection
data_all = zeros(length(suffix_list), numEpoch*2, length(seed_list));
data_second = zeros(length(suffix_list), numEpoch, length(seed_list));

for ii = 1:length(suffix_list)
    for jj = 1:length(seed_list)
        data_tmp = [];

        suffix = suffix_list{ii};
        seed = seed_list(jj);

        result1 = load([pathResult_simul num2str(seed) 'Trained_result_' name1 '_rgb' suffix '.mat']);
        result2 = load([pathResult_simul num2str(seed) 'Trained_result_' name1 '_' name2 '_rgb' suffix '.mat']);

        len = length(result1.result.ValidationAccuracy(:, 2));
        step = floor(len/numEpoch);
        data_tmp = result1.result.ValidationAccuracy(1:step:step*numEpoch-1, 2)';

        len = length(result2.result.ValidationAccuracy2(:, 2));
        step = floor(len/numEpoch);
        data_tmp = [data_tmp result2.result.ValidationAccuracy2(1:step:step*numEpoch-1, 2)'];
        
        data_all(ii, :, jj) = data_tmp;
        data_second(ii, :, jj) = result2.result.ValidationAccuracy1(1:step:step*numEpoch-1, 2)';
    end
end

%%% boxplot
figure;
data_bar = squeeze(data_all(:, end, :));
x = 1:2:2*length(suffix_list);
mj_boxplot(data_bar', palette_network_tmp);
hold on

set(gca, 'tickdir', 'out'); box off
ylim([0 60])
yticks(0:20:60)
ylabel('final accuracy (%)')
yline(15.2515, '--', 'Color', [.33 .33 .33], 'LineWidth', 2)
mj_plotctrl([30 30])
exportgraphics(gcf, 'F2/PS_filtcontrol.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none')

%% Fig. 2g - final accuracy for each pair
%%% parameters
domain_name = {'photo', 'art_painting', 'cartoon', 'sketch'};
pairs = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4;4 3; 4 2; 4 1; 3 2; 3 1; 2 1];

suffix_list = {'', '_gabor_V1_0.8'};
suffix_name = {'DNN', 'GbDNN'};

seed_list = 1:20;

%%% data collection
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

%%% Plotting - top panel 
figure; hold on
data_diff = squeeze(data_all(2, :, :) - data_all(1, : ,:));
[~, ind_ref] = sort(mean(data_diff, 2), 'descend');

data_all_srt = data_all(:, ind_ref, :);
pairs_srt = pairs(ind_ref, :);

for ii = 1:length(suffix_list)
    plot(1:size(pairs, 1), mean(squeeze(data_all_srt(ii, :, :))', 1), 'Color', palette_network(ii, :), 'LineWidth', 1.5);
    plot(1:size(pairs, 1), mean(squeeze(data_all_srt(ii, :, :))', 1), '.',  'Color', palette_network(ii, :), 'MarkerSize', 10);
    set(gca, 'tickdir', 'out')
    xticks(1:size(pairs, 1))
    xtick_labels = cell(1, size(pairs, 1));
    pairs_srt = pairs(ind_ref, :);
    for pp = 1:length(pairs_srt)
        xtick_labels{pp} = [upper(domain_name{pairs_srt(pp, 1)}(1)) upper(domain_name{pairs_srt(pp, 2)}(1))];
    end
    set(gca,'XTickLabel',xtick_labels)
    xlim([0 size(pairs, 1)+1]); ylim([10 80]); yticks(10:20:70); 
    ylabel('accuracy (%)'); 
    xlabel('domain pairs');
    box off
    mj_plotctrl([45 14])
end
exportgraphics(gcf, ['F2/Alldomain_acc_DNN_GbDNN.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none')

%%% Plotting - bottom panel
f = figure;
f.Position = [300 500 267 178];
data_diff = data_diff(ind_ref, :);

mj_shadedplot(1:size(pairs, 1), data_diff, palette_domain(3, :)); hold on
plot(1:size(pairs, 1), mean(data_diff, 2), '.',  'Color', palette_domain(3, :), 'MarkerSize', 10);

set(gca, 'tickdir', 'out')
xticks(1:size(pairs, 1))
xtick_labels = cell(1, size(pairs, 1));
for pp = 1:size(pairs, 1)
    xtick_labels{pp} = [upper(domain_name{pairs_srt(pp, 1)}(1)) upper(domain_name{pairs_srt(pp, 2)}(1))];
end
set(gca,'XTickLabel',xtick_labels)
xlim([0 size(pairs, 1)+1]); ylim([-10 30]); yticks(-10:10:30); ylabel(''); xlabel('domain pairs')
box off
mj_plotctrl([45 14])
exportgraphics(gcf, 'F2/Alldomain_acc_diff.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none')

%%% statistical testing
for pp = 1:size(pairs, 1)
    data1 = data_all_srt(1, pp, :); data2 = data_all_srt(2, pp, :);
    [h, p] = ttest(data1, data2);

    if p < 0.05 && mean(data1) > mean(data2)
        msg = [suffix_name{1} ' better'];
    elseif p < 0.05 && mean(data1) < mean(data2)
        msg = [suffix_name{2} ' better'];
    else
        msg = "non isg";
    end

    fprintf("For %s pair, %s p = %.2d\n", xtick_labels{pp}, msg, p)
end

finalacc_diff_all = data_diff;

%% Fig. 2f - all pairs
figure;
mj_boxplot_dot_line_verMean(data_all_srt(:, :, :), palette_network)
ylim([10 80]); yticks(10:20:80); 
ylabel('accuracy (%)'); 
box off
mj_plotctrl([15 30])
exportgraphics(gcf, ['F2/Alldomain_acc_summary_boxdotline.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none')

%% Simulation - Image PCA
name_domain = {'photo', 'art_painting', 'cartoon', 'sketch'};
file_tot = [];
class_tot = [];
domain_tot = [];

for dd = 1:length(name_domain)
    name_tmp = name_domain{dd};
    path = [path0 '/data/images/pacs/' name_tmp];
    imds = imageDatastore(path,"FileExtensions",[".jpg", ".png", ".JPEG"], "IncludeSubfolders",true, ...
        "LabelSource","foldernames");
    labels = zeros(length(imds.Labels), 1);
    labels_name = string(unique(imds.Labels));

    for ii = 1:length(labels_name)
            labels(string(imds.Labels) == labels_name(ii)) = ii;
    end
    imds.Labels = labels;

    file_tot = [file_tot; imds.Files];
    class_tot = [class_tot; imds.Labels];
    domain_tot = [domain_tot; ones(length(imds.Labels), 1) * dd];
end

% image gathering
img_tot = zeros(length(file_tot), 227*227*3);

parfor ii = 1:length(file_tot)
    img = imread(file_tot{ii});
    img = im2double(img);

    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end
    img = rescale(img);
    img_tot(ii, :) = img(:)';
end

% image PCA
[~, img_pca, ~, ~, explained] = pca(img_tot);
save([pathResult_analysis 'PACS_PCA_score.mat'], 'img_pca')
save([pathResult_analysis 'PACS_PCA_explained.mat'], 'explained')

%% Fig. 2h
load([pathResult_analysis 'PACS_PCA_score.mat'])
load([pathResult_analysis 'PACS_PCA_explained.mat'])

f = figure;
for ii = 1:length(unique(domain_tot))
    scatter(img_pca(domain_tot==ii, 1), img_pca(domain_tot==ii, 2), ...
        5, 'MarkerFaceColor', palette_domain(ii, :), 'MarkerEdgeColor', ...
        'none', 'MarkerFaceAlpha', 0.1)
    hold on 
end
xlim([-300 200]); xticks([-300:100:200])
ylim([-150 150]); yticks([-150:100:150])
xlabel('PC1 (50.17%)'); ylabel('PC2 (3.70%)')
mj_plotctrl([30 30])
exportgraphics(gcf, ['F2/image_pca.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none')

%% Simulation - image distance measure
load([pathResult_analysis 'PACS_PCA_score.mat'])
load([pathResult_analysis 'PACS_PCA_explained.mat'])
name_domain = {'photo', 'art_painting', 'cartoon', 'sketch'};

% Collect image metadata
file_tot = []; class_tot = []; domain_tot = [];
for dd = 1:length(name_domain)
    % extract datas
    name_tmp = name_domain{dd};
    path = [path0 '/data/images/pacs/' name_tmp];
    imds = imageDatastore(path,"FileExtensions",[".jpg", ".png", ".JPEG"], "IncludeSubfolders",true, ...
        "LabelSource","foldernames");
    labels = zeros(length(imds.Labels), 1);
    labels_name = string(unique(imds.Labels));

    for ii = 1:length(labels_name)
            labels(string(imds.Labels) == labels_name(ii)) = ii;
    end
    imds.Labels = labels;

    % variables update
    file_tot = [file_tot; imds.Files];
    class_tot = [class_tot; imds.Labels];
    domain_tot = [domain_tot; ones(length(imds.Labels), 1) * dd];
end

distances = zeros(4, 4);
d_kld = zeros(4, 4);
d_jsd = zeros(4, 4);
pca_data = img_pca;

exp_all = cumsum(explained);
thr = find(exp_all > 99, 1);
pca_data = img_pca(:, 1:thr);

% distance measure
for ii = 1:4
    for jj = 1:4
        if ii == jj
            continue
        else
            inds1 = domain_tot == ii; inds2 = domain_tot == jj;
            
            distances(ii, jj) = kl_divergence_knn(pca_data, inds1, inds2);
        end
    end
end
save([pathResult_analysis 'dist_knn_k10.mat'], 'distances')

%% Fig. 2i
ver = 'knn_k10';
data_diff = squeeze(data_all_srt(2, :, :) - data_all_srt(1, :, :));
load([pathResult_analysis 'dist_' ver '.mat']);
distances = distances / 1e3;
figure;
d_list = zeros(1, 12);
for ii = 1:length(d_list)
    d_list(ii) = distances(pairs_srt(ii, 1), pairs_srt(ii, 2));
end
y = mean(data_diff, 2);
y_err = std(data_diff, [], 2);

hold on

scatter(d_list(1:6), mean(data_diff(1:6, :), 2), 10, 'o',  ...
    'MarkerFaceColor', palette_domain(4, :), 'MarkerEdgeColor', 'none'); hold on
errorbar(d_list(1:6), y(1:6), y_err(1:6)', 'vertical', 'CapSize', 0, ...
    'LineWidth', 0.5', 'Color', palette_domain(4, :), 'LineStyle','none')

scatter(d_list(7:12), mean(data_diff(7:12, :), 2), 10, 'o',  ...
    'MarkerFaceColor', palette_domain(3, :), 'MarkerEdgeColor', 'none'); hold on
errorbar(d_list(7:12), y(7:12), y_err(7:12)', 'vertical', 'CapSize', 0, ...
    'LineWidth', 0.5', 'Color', palette_domain(3, :), 'LineStyle','none')

xlabel('distance')
ylabel('dACC')
title('dACC vs. domain distance')
hold on
xlim([0 5]); ylim([-10 35])
xticks(0:1:5); yticks([-10:10:30])
x = linspace(0, 5, 10);
mdl = fitlm(d_list, mean(data_diff, 2));
plot(x, x * mdl.Coefficients.Estimate(2) + mdl.Coefficients.Estimate(1), '--', 'Color', [.5 .5 .5])
mj_plotctrl([30 30])

[r, p] = corrcoef(d_list, mean(data_diff, 2)')
exportgraphics(gcf, ['F2/d_vs_dacc_corr.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none')

%% Fig. 2i - inset
ver = 'knn_k10';
load([pathResult_analysis 'dist_' ver '.mat']);

fig_size = [10 10];

%%% Plotting top panel
dist_srt = [];
for ii = 1:size(pairs, 1)
    dist_srt(ii) = distances(pairs_srt(ii, 1), pairs_srt(ii, 2));
end

figure;
dist_bar = [dist_srt(1:6); dist_srt(7:12)];
mj_boxplot(dist_bar' / 1000, [palette_domain(3, :); palette_domain(3, :)]);
ylabel('KLD')
ylim([0 5]); yticks([0 5]);
mj_plotctrl(fig_size)
exportgraphics(gcf, ['F2/distances_compare.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none')
[h, p] = ranksum(dist_bar(1, :), dist_bar(2, :))

%%% Plotting bottom panel
figure;
data_diff = squeeze(data_all_srt(2, :, :) - data_all_srt(1, :, :));
data_diff_bar = [reshape(data_diff(1:6, :), 1, []); reshape(data_diff(7:12, :), 1, [])];
mj_boxplot(data_diff_bar', [palette_domain(3, :); palette_domain(3, :)]);
ylim([-10 40])
yticks([-10 40])
ylabel('dacc')
mj_plotctrl(fig_size)
exportgraphics(gcf, ['F2/dacc_compare.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none')
[h, p] = ranksum(data_diff_bar(1, :), data_diff_bar(2, :))

%% Subfunctions
function kl_estimate = kl_divergence_knn(pca_data, inds1, inds2)
    % A: MxP matrix where M is the number of samples from distribution P
    % B: NxP matrix where N is the number of samples from distribution Q
    % k: the number of nearest neighbors (default is 1)
    A = pca_data(inds1, :);
    B = pca_data(inds2, :);

    k = 10;
    
    % Number of samples from A and B
    [n, d] = size(A);
    [m, ~] = size(B);
    
    % Step 1: Compute nearest neighbors within A (for P distribution)
    [idx_A_to_A, dist_A_to_A] = knnsearch(A, A, 'K', k + 1); % k+1 to avoid self-distance
    dist_A_to_A = dist_A_to_A(:, k+1); % Ignore the self-distance (first column)

    % Step 2: Compute nearest neighbors from A to B (for Q distribution)
    [idx_A_to_B, dist_A_to_B] = knnsearch(B, A, 'K', k); % Nearest neighbors from A to B
    dist_A_to_B = dist_A_to_B(:, k); % Ignore the self-distance (first column)

    % Step 3: Compute log ratio of nearest neighbor distances
    log_ratio = log(dist_A_to_B ./ dist_A_to_A);
    
    % Step 4: Average over all samples to get the KL divergence estimate
    kl_estimate = (d/n) * sum(log_ratio) + log(m / (n-1));
end