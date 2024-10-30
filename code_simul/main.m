%% settings
clear; clc; close;

path0 = 'your_project_folder';
cd([path0 '\code_simul'])
addpath([path0 '\code_basic'])
basicSettings;

%% basic parameters - do not modify 
name = {'pacs\photo', 'pacs\sketch'};

% hyperparameters 
mode.numChannel = 3; % 3 for rgb, 1 for gray
mode.learningRate = 1e-4;
mode.batchSize = 64;
mode.numEpoch = 90;
mode.visualize = 'off';
mode.simulation = 0; % 0 for all, 1 for first, 2 for second training
mode.portion = 1; % portion of data to use
mode.parameterSearch = 0;
mode.fix = 0;
mode.convfix = 0;
mode.lrdropfactor = 0.1;
mode.lrdropperiod = 30;
mode.inputProcess = 'rescale';
mode.w_initializer = 'he';
mode.b_initializer = 'zeros';
mode.numClass = 7;

% simulation conditions
suffix_list = {'', '_gabor_V1_0.8'};
seed_list = 1:5;

%% Simulations
for seed = seed_list
    tic

    for ii = 1:length(suffix_list)
        suffix = suffix_list{ii};
        
        % setting random generation
        rng(seed)
        
        % load initial network 
        net = initAlexnet([227 227 mode.numChannel], mode.numClass, [29 29], mode.convfix, mode.inputProcess, ...
           mode.w_initializer, mode.b_initializer);
        
        % add filters or not
        if contains(suffix, 'gabor') || contains(suffix, 'gau') || ...
                contains(suffix, 'dog') || contains(suffix, 'comb')
            net = addfilter([pathData 'filter parameters\'], net, suffix);
            disp("Filter added!")
        end

        % if randomly fixed networks
        if contains(suffix, 'gabor') || contains(suffix, 'gau') || ...
                contains(suffix, 'dog') || contains(suffix, 'comb') || contains(suffix, 'rand')
            mode.fix = 1;
        else
            mode.fix = 0;
        end

        % if gabor with nonfix
        if contains(suffix, 'nofix')
            mode.fix = 0;
        end

        simul_sequential(seed, net, name, suffix, mode)
    end
    toc
end





