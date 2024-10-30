function [] = simul_sequential(seed, name, suffix, mode)
%%%% Sequential training file
%% path setting
basicSettings;
rng(seed);


%% load network
filter_size = [30 30];

net = initAlexnet([227 227 mode.numChannel], mode.numClass, filter_size, mode.convfix, mode.inputProcess, ...
   mode.w_initializer, mode.b_initializer);

% add filters or not
if contains(suffix, 'gabor') || contains(suffix, 'gau') || ...
        contains(suffix, 'dog') || contains(suffix, 'comb')
    net = addfilter([pathData 'filter parameters/'], net, suffix, seed);
    mode.fix = 1;
    disp("Filter added and fixed!")
end

if contains(suffix, 'unfix')
    mode.fix = 0;
    disp("Unfixing version")
elseif contains(suffix, 'fix')
    mode.fix = 1;
    disp("Fixing version")
end


%% parameters
numChannel = mode.numChannel;
learningRate = mode.learningRate;
batchSize = mode.batchSize;
visualize = mode.visualize;
numEpoch = mode.numEpoch;
portion = mode.portion;
fix = mode.fix;
convfix = mode.convfix;
LearnRateDropFactor = mode.lrdropfactor;
LearnRateDropPeriod = mode.lrdropperiod;
numClass = mode.numClass;

% rgb or gray images
if numChannel == 3
    suffix0 = 'rgb';
elseif numChannel == 1
    suffix0 = 'gray';
end

% number of filters
if net.Layers(2,1).NumFilters ~= 96
    suffix0 = [suffix0 sprintf('_nfilt%d', net.Layers(2,1).NumFilters)];
end

% proportion
if portion ~= 1
    suffix0 = [suffix0 '_' num2str(portion)];
end

% if convfix
if mode.convfix
    suffix0 = [suffix0 '_convfix'];
end

% name settings
name_true = cell(1, length(name));
for ii = 1:length(name)
    if contains(name{ii}, 'pacs')
        name_true{ii} = name{ii}(6:end);
    elseif contains(name{ii}, 'officehome')
        name_true{ii} = name{ii}(12:end);
    elseif contains(name{ii}, 'cifar-10')
        name_true{ii} = name{ii}(1:8);
    elseif contains(name{ii}, 'domainnet/train')
        name_true{ii} = name{ii}(17:end);
    elseif contains(name{ii}, 'ILSVRC')
        name_true{ii} = 'imagenet';
    else
        name_true{ii} = name{ii};
    end
end

%% extract images and model
% data generation
train = cell(1, length(name));
valid = cell(1, length(name));

% data for sequential training
for i = 1:length(name)
    % % extract imagedatastore
    disp([pathData_img name{i}])
    imds = extractImg(pathData_img, name{i}, portion);
    imds.Labels = categorical(imds.Labels);

    % split training/validation
    [imdsTrain,imdsValidation] = splitEachLabel(imds,0.9,'randomized');

    % image augmentation
    pixelRange = [-30 30];
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection', true, ...
        'RandXTranslation',pixelRange, ...
        'RandYTranslation',pixelRange);

    % color processing
    if numChannel == 3
        option = 'gray2rgb';
    elseif numChannel == 1
        option = 'rgb2gray';
    end

    augimdsTrain = augmentedImageDatastore([227 227],imdsTrain, 'ColorPreprocessing',option,...
        'DataAugmentation',imageAugmenter);
    augimdsValidation = augmentedImageDatastore([227 227],imdsValidation, 'ColorPreprocessing',option);

    train{i} = augimdsTrain; valid{i} = augimdsValidation;
end

%% Main simulations
disp("SEQUENTIAL TRAINING")
fprintf("Seed = %d\n", seed)
disp(['Suffix = ' suffix])
fprintf("Images = ")
disp(name)

if mode.simulation == 1 || mode.simulation == 0
    %%%%%%%% Phase 1 training %%%%%%%%
    %% Main trainings - DNN vs. DNN + Gabor (3 types)    
    suffix_all = [suffix0 suffix '.mat'];
    
    % Training 1
    [net1, result1] = trainAlexnet(net, train{1}, valid{1}, fix, convfix,  ...
        learningRate, LearnRateDropFactor, LearnRateDropPeriod, batchSize, numEpoch, visualize , 1, 'first');
    
    result1.learningRate = learningRate; result1.batchSize = batchSize; result1.maxEpoch = numEpoch;
    
    net = net1; result = result1;

    save([pathResult_simul_save num2str(seed) 'Trained_nn_' name_true{1} '_' suffix_all], 'net');
    save([pathResult_simul_save num2str(seed) 'Trained_result_' name_true{1} '_' suffix_all], 'result');
            
    fprintf('\n')
    disp("TRAINING 1 DONE!")
    fprintf('\n')
end

if mode.simulation == 2 || mode.simulation == 0
    %%%%%%%% Phase 2 training %%%%%%%%
    %% Main trainings - DNN vs. DNN + Gabor (original, high, low)
    % DNN / DNN + Gabor
    suffix_all = [suffix0 suffix '.mat'];

    load([pathResult_simul_save num2str(seed) 'Trained_nn_' name_true{1} '_' suffix_all]);
    
    % Training 2
    [net2, result2] = trainAlexnet(net, train{2}, valid{2}, fix, convfix, ...
        learningRate, LearnRateDropFactor, LearnRateDropPeriod, batchSize, numEpoch, visualize, 1, 'second', valid{1});
    
    result2.learningRate = learningRate; result2.batchSize = batchSize; result2.maxEpoch = numEpoch;
    
    net = net2; result = result2;

    save([pathResult_simul_save num2str(seed) 'Trained_nn_' name_true{1} '_' name_true{2} '_' suffix_all], 'net');
    save([pathResult_simul_save num2str(seed) 'Trained_result_' name_true{1} '_' name_true{2} '_' suffix_all], 'result');
    
    fprintf('\n')
    disp("TRAINING 2 DONE!")
    fprintf('\n')
end
end
