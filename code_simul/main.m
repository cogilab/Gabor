%% settings
clear; clc; close;

path0 = 'your-project-path';
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
        simul_sequential(seed, name, suffix, mode)
    end
    toc
end