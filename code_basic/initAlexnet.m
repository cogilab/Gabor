function [net] = initAlexnet(size_in, size_out, size_filter, convfix, inputProcess, w_initializer, b_initializer)
%% initial network
layers = alexnet('Weights', 'none');

% change first layer into the customized size
if strcmp(inputProcess, 'rescale')
    inputLayer = imageInputLayer(size_in, 'Name', 'data', 'Normalization', 'rescale-zero-one', ...
        'Max', 255', 'Min', 0);
elseif strcmp(inputProcess, 'zerocenter')
    inputLayer = imageInputLayer(size_in, 'Name', 'input', 'Normalization', 'zerocenter', 'Mean', ones(1, 1, 3) * (255/2));
else
    error("Invalid Input Processing Option! (mode.inputProcess)")
end

% conv1 layer with customized filter size
conv1Layer = convolution2dLayer(size_filter, 96, 'Name', 'conv1', 'NumChannels', size_in(3), ...
'Stride', [4 4], 'Weights', [], 'Bias', [], 'WeightsInitializer', w_initializer);

% change classification layer into customized size
fc8Layer = fullyConnectedLayer(size_out, 'Name', 'fc8');

% Assemble
newlayers = [
    inputLayer
    conv1Layer
    layers(3:22)
    fc8Layer
    softmaxLayer];

% if convfix -> network = 5 convolutional layers + 1 fc layer
if convfix
    fc6Layer = fullyConnectedLayer(size_out, 'Name', 'fc6');
    newlayers = [
        inputLayer
        conv1Layer
        layers(3:16)
        fc6Layer
        softmaxLayer];
end

% weight initialization
for i = 1:numel(newlayers)
    if isa(newlayers(i), 'nnet.cnn.layer.Convolution2DLayer') || ...
            isa(newlayers(i), 'nnet.cnn.layer.FullyConnectedLayer') || ...
            isa(newlayers(i), 'nnet.cnn.layer.GroupedConvolution2DLayer')
        newlayers(i).WeightsInitializer = w_initializer;
    end
end

% bias initialization
layer_tmp = [6 12 14 17 20 23];
for ii = 1:length(layer_tmp)
    newlayers(layer_tmp(ii)).BiasInitializer = b_initializer;
end

% returning
net = dlnetwork(newlayers, Initialize=true);

