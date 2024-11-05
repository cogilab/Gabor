function [net, result] = trainAlexnet(net, augimdsTrain, augimdsValid, fix, convfix, learningRate, LearnRateDropFactor, LearnRateDropPeriod, batchSize, numEpoch, visualize, isshuffle, varargin)
%% AlexNet Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. net: network of class 'dlnetwork
% 2. augimdsTrain: augmented training imds
% 3. augimdsValid: augmented validation imds
% 4. fix: whether fixing conv1 layer's weights
% ~varargin{1}: if 'second' -> MNIST training version
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Manual training
% for ADAM
averageGrad = [];
averageSqGrad = [];

% for SGDM
vel = [];
momentum = 0.9;

% L2 regularization factor
param_L2 = 0;

% idxWeigths for learnable parameters
idxWeights = ismember(net.Learnables.Parameter,["Weights" "Scale"]);

% fixing the first layer or not
if fix
    disp("conv1 fixed")
    layer = net.Layers(2, 1);
    layer.WeightLearnRateFactor = 0;
    net = replaceLayer(net, 'conv1', layer);
    idxWeights(1) = 0;
end
if convfix
    disp("all conv fixed")
    jj = 1;
    for ii = [2 6 10 12 14]
        layer = net.Layers(ii, 1);
        layer.WeightLearnRateFactor = 0;
        net = replaceLayer(net, sprintf('conv%d', jj), layer);
        idxWeights(1) = 0;
        jj = jj + 1;
    end
end
net = dlnetwork(net.Layers);

% parameters
numObservations = augimdsTrain.NumObservations;
numIterationsPerEpoch = floor(numObservations./batchSize);
numIteration = numEpoch * numIterationsPerEpoch;
validationFrequency = numIterationsPerEpoch;

% monitor setting
monitor = trainingProgressMonitor;
monitor.Info = ["LearningRate","Epoch","Iteration"];
monitor.XLabel = "Iteration";
monitor.Visible = visualize;

% error handling
if strcmp(augimdsTrain.ColorPreprocessing, 'rgb2gray')
    batch_set = 'SSBC';
else
    batch_set = 'SSCB';
end

if strcmp(varargin{1}, 'first')
    monitor.Metrics = ["TrainingLoss", "ValidationLoss", "ValidationAccuracy"];
    groupSubPlot(monitor,"Loss", ["TrainingLoss", "ValidationLoss"]);
    groupSubPlot(monitor,"Accuracy", "ValidationAccuracy");
elseif strcmp(varargin{1}, 'second')
    monitor.Metrics = ["TrainingLoss", "ValidationLoss1", "ValidationLoss2", ...
        "ValidationAccuracy1", "ValidationAccuracy2"];
    groupSubPlot(monitor,"Loss", ["ValidationLoss1", "ValidationLoss2", "TrainingLoss"]);
    groupSubPlot(monitor,"Accuracy", ["ValidationAccuracy1", "ValidationAccuracy2"]);
end

% make minibatch queue
mbqTrain = minibatchqueue(augimdsTrain,...
    'MiniBatchSize',batchSize,...
    'PartialMiniBatch', 'discard',...
    'OutputAsDlarray',[1,0],...
    'MiniBatchFormat',{batch_set,''},...
    'OutputEnvironment',{'gpu','cpu'});

mbqValid = minibatchqueue(augimdsValid,...
    'MiniBatchSize',batchSize,...
    'OutputAsDlarray',[1,0],...
    'MiniBatchFormat',{batch_set,''},...
    'OutputEnvironment',{'gpu','cpu'});

% shuffle data
if isshuffle
    shuffle(mbqValid);
end

if strcmp(varargin{1}, 'second')
    mbqValid2 = minibatchqueue(varargin{2},...
    'MiniBatchSize',batchSize,...
    'OutputAsDlarray',[1,0],...
    'MiniBatchFormat',{batch_set,''},...
    'OutputEnvironment',{'gpu','cpu'});
    numValid2 = varargin{2}.NumObservations;
end

accfun = dlaccelerate(@modelLoss);
clearCache(accfun);
numValid = augimdsValid.NumObservations;

%%% Training loop
iteration = 0; epoch = 0; lossValidation = 0; accuracyValidation = 0;

while epoch < numEpoch && ~monitor.Stop
    epoch = epoch + 1;
    
    if rem(epoch, LearnRateDropPeriod) == 0
        learningRate = learningRate * LearnRateDropFactor;
    end
    
    % reset data
    reset(mbqTrain);

    % shuffle data
    if isshuffle
        shuffle(mbqTrain);
    end

    while hasdata(mbqTrain) && ~monitor.Stop
        iteration = iteration + 1;

        %%% Training & Update
        [X,Y] = next(mbqTrain);

        % training acc / loss
        [lossTrain,gradients,state] = dlfeval(accfun,net,X,Y,idxWeights,param_L2);
        net.State = state;

        % Update the network parameters using the SGDM optimizer.
        [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iteration,...
            learningRate);
        
        %%% validation acc / loss
        if rem(iteration-1, validationFrequency) == 0
            % for first validation
            reset(mbqValid);
            
            % calculate validation accuracy
            accuracyValidation = 0; lossValidation = 0;
            while hasdata(mbqValid)
                [X, Y] = next(mbqValid);
                
                % loss, acc calculation
                [loss_temp, acc_temp] = modelLossValid(net,X,Y);
                accuracyValidation = accuracyValidation + acc_temp * length(Y);
                lossValidation = lossValidation + loss_temp * length(Y);
            end
            lossValidation = lossValidation / numValid;
            accuracyValidation = accuracyValidation * 100 / numValid;

            % for second validation
            if strcmp(varargin{1}, 'second')
                % first validation saving
                lossValidation1 = lossValidation;
                accuracyValidation1 = accuracyValidation;

                % for first validation
                reset(mbqValid2);
    
                accuracyValidation2 = 0; lossValidation2 = 0;
                while hasdata(mbqValid2)
                    [X, Y] = next(mbqValid2);
                
                    % loss, acc calculation
                    [loss_temp, acc_temp] = modelLossValid(net,X,Y);
                    accuracyValidation2 = accuracyValidation2 + acc_temp * length(Y);
                    lossValidation2 = lossValidation2 + loss_temp * length(Y);
                end
                lossValidation2 = lossValidation2 / numValid2;
                accuracyValidation2 = accuracyValidation2 * 100 / numValid2;
            end

            fprintf("epoch = %d / acc = %.2f\n", epoch, accuracyValidation);
            if strcmp(varargin{1}, 'second')
                fprintf("(2) epoch = %d / acc = %.2f\n", epoch, accuracyValidation2);
            end
        end
        
        
        % Update the training progress monitor.
        updateInfo(monitor, ...
            Epoch=string(epoch) + " of " + string(numEpoch), ...
            Iteration=string(iteration) + " of " + string(numIteration));

       if strcmp(varargin{1}, 'first')
           recordMetrics(monitor,iteration, ...
            TrainingLoss=lossTrain, ...
            ValidationLoss=lossValidation, ...
            ValidationAccuracy=accuracyValidation);
       elseif strcmp(varargin{1}, 'second')
           recordMetrics(monitor,iteration, ...
            TrainingLoss=lossTrain, ...
            ValidationLoss1=lossValidation1, ...
            ValidationLoss2=lossValidation2, ...
            ValidationAccuracy1=accuracyValidation1, ...
            ValidationAccuracy2=accuracyValidation2);
       end
       
       monitor.Progress = 100*iteration/numIteration;
    end
end
result = monitor.MetricData;

function [loss,gradients, state] = modelLoss(net,X,Y,idxWeights,param_L2)
    % fc index
    final_fc_indx = length(net.Layers);

    % with/without dropouts
    [YPred, state] = forward(net,X);   
    
    % loss calculation
    Y = onehotencode(Y, 2, 'ClassNames', 1:net.Layers(final_fc_indx-1, 1).OutputSize)';
    loss = crossentropy(YPred,Y);    
    
    % gradients
    gradients = dlgradient(loss,net.Learnables);  

    % L2 loss
    gradients(idxWeights, :) = ...
        dlupdate(@(g, w) g + 0.5 * param_L2 * w, gradients(idxWeights, :), net.Learnables(idxWeights, :));
end

function [loss,accuracy] = modelLossValid(net,X,Y)
    % fc index
    final_fc_indx = length(net.Layers);

    % with/without dropouts
    YPred = predict(net,X);   

    % acc calculation
    YPred_temp = single(onehotdecode(YPred, 1:net.Layers(final_fc_indx-1, 1).OutputSize, 1)');
    accuracy = sum(YPred_temp == Y) / length(Y);

    % loss calculation
    Y = onehotencode(Y, 2, 'ClassNames', 1:net.Layers(final_fc_indx-1, 1).OutputSize)';
    loss = crossentropy(YPred,Y);
end
end


