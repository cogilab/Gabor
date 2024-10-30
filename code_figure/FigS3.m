%% Desciprions
% This script is to reproduce figures from supplementary Figure 3.
% If you want to obtain data, you first need to visit "code_simul" folder.

%% path settings
clear; close; clc;
path0 = '/home/dgxadmin/Minjun/Project-DNN_Gabor';
cd([path0 '/figures'])
addpath([path0 '/code_basic'])
basicSettings;

%% Fig. S3 a-b
%%% parameters
pathResult_simul = [path0 '/results/results_simulation/'];
domain_name = {'photo', 'art_painting', 'cartoon', 'sketch'};
pairs = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4; 4 3; 4 2; 4 1; 3 2; 3 1; 2 1];

suffix_list = {'', '_gabor_V1_0.8'};

seed_list = 1:20;
numEpoch = 90;
name_all = {'photo', 'art', 'cartoon', 'sketch'};

%%% data collection
data_all_first = zeros(length(suffix_list), length(domain_name), numEpoch, length(seed_list));
data_all_second = zeros(length(suffix_list), size(pairs, 1), numEpoch, length(seed_list));

for ii = 1:length(suffix_list)
    for jj = 1:length(seed_list)
        for dd = 1:length(domain_name)
            suffix = suffix_list{ii};
            seed = seed_list(jj);
            name = domain_name{dd};

            result1 = load([pathResult_simul num2str(seed) 'Trained_result_' name '_rgb' suffix '.mat']);
            
            data_tmp = result1.result.ValidationAccuracy(:, 2);
            len = length(data_tmp); step = floor(len / numEpoch);

            data_all_first(ii, dd, :, jj) = data_tmp(1:step:step*numEpoch);
        end
        for pp = 1:size(pairs, 1)
            name1 = domain_name{pairs(pp, 1)};
            name2 = domain_name{pairs(pp, 2)};

            result2 = load([pathResult_simul num2str(seed) 'Trained_result_' name1 '_' name2 '_rgb' suffix '.mat']);
            
            data_tmp = result2.result.ValidationAccuracy1(:, 2);
            len = length(data_tmp); step = floor(len / numEpoch);

            data_all_second(ii, pp, :, jj) = data_tmp(1:step:step*numEpoch);
        end
    end
end

%%% plotting
for ii = 1:length(suffix_list)
    for dd = 1:length(domain_name)
        figure;
        data_all = squeeze(data_all_first(ii, dd, :, :));
        mj_shadedplot(1:numEpoch, data_all, palette_network(ii, :), 3)
        hold on; data_all_save = data_all;

        data_all = squeeze(mean(data_all_second(ii, pairs(:, 2) == dd, :, :), 2));
        mj_shadedplot(1:numEpoch, data_all, palette_network(ii, :) * 0.8, 3)

        box off;
        xticks(1:45:90); xlim([1 90]); ylim([0 100]); yticks([0 50 100]);
        xticklabels({'', '', ''}); yticklabels({'', '', ''})
        set(gca, 'tickdir', 'out')
        hold on
        plot(1:numEpoch, ones(1, numEpoch) * (100/7), '--', 'Color', [.33 .33 .33], 'LineWidth', 1)
        
        [h, p] = ttest(data_all_save(end, :), data_all(end, :));    
        fprintf("%d ii and %d dd, p = %.2f\n", ii, dd, p)
        mj_plotctrl([20 20])
        exportgraphics(gcf, ['SF/Training_1st_2nd_' name_all{dd} suffix_list{ii} '.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none')
    end
end


%% Fig. S3 c-d
%%% parameters
pathResult_simul = [path0 '/results/results_simulation/'];
domain_name = {'photo', 'art_painting', 'cartoon', 'sketch'};
pairs = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4; 4 3; 4 2; 4 1; 3 2; 3 1; 2 1];

suffix_list = {'', '_gabor_V1_0.8'};
suffix_name = {'DNN', 'GbDNN'};

seed_list = 1:20;
numEpoch = 90;

%%% data collection
data_all_first = zeros(length(suffix_list), length(domain_name), numEpoch, length(seed_list));
data_all_second = zeros(length(suffix_list), size(pairs, 1), numEpoch, length(seed_list));

for ii = 1:length(suffix_list)
    for jj = 1:length(seed_list)
        for dd = 1:length(domain_name)
            suffix = suffix_list{ii};
            seed = seed_list(jj);
            name = domain_name{dd};

            result1 = load([pathResult_simul num2str(seed) 'Trained_result_' name '_rgb' suffix '.mat']);
            
            data_tmp = result1.result.ValidationAccuracy(:, 2);
            len = length(data_tmp); step = floor(len / numEpoch);

            data_all_first(ii, dd, :, jj) = data_tmp(1:step:step*numEpoch);
        end
        for pp = 1:size(pairs, 1)
            name1 = domain_name{pairs(pp, 1)};
            name2 = domain_name{pairs(pp, 2)};

            result2 = load([pathResult_simul num2str(seed) 'Trained_result_' name1 '_' name2 '_rgb' suffix '.mat']);
            
            data_tmp = result2.result.ValidationAccuracy1(:, 2);
            len = length(data_tmp); step = floor(len / numEpoch);

            data_all_second(ii, pp, :, jj) = data_tmp(1:step:step*numEpoch);
        end
    end
end

%%% plotting
for ii = 2
    for jj = 1:4
        figure(jj)
    end
    
    for pp = 1:size(pairs, 1)
        figure(pairs(pp, 2))

        data_all = squeeze(data_all_second(ii, pp, :, :));
        plot(1:numEpoch, mean(data_all'),'Color', palette_domain(pairs(pp, 1), :), 'LineWidth', 1.5)
        box off;
        xticks([1 45 90]); xlim([1 90]); ylim([0 100]); yticks([0 50 100]);
        xticklabels({'', '' ,''}); yticklabels({'', '', ''})
        set(gca, 'tickdir', 'out')
        hold on
        plot(1:numEpoch, ones(1, numEpoch) * (100/7), '--', 'Color', [.349 .349 .349], 'LineWidth', 1)
        
    end

    for jj = 1:4
        figure(jj)
        mj_plotctrl([20 20])
        exportgraphics(gcf, ['SF/Training_2nd_' name_all{jj} suffix_list{ii} '.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none')
    end
end