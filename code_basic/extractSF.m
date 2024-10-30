function [X, Y, filepath] = extractSF(path0, name, power, varargin)
%% Exract Spatial Frequency of Images
% input: 
% path0 = project folder
% name = dataset name e.g. 'mnist'
% prefix = 'r' for resized, '' for original
% ntot = total # of images
% nclass= total # of classes
% is_only_filepath = 1 if only filepath returning
% output: mean distribution of spatial frequency, filepath
switch name
    case {'pacs\photo', 'pacs\art_painting', 'pacs\cartoon', 'pacs\sketch', ...
            'domainnet\train\real', 'domainnet\train\quick', 'domainnet\train\sketch', 'domainnet\train\info', 'domainnet\train\clip', 'domainnet\train\paint'}
        imds = extractImg(path0, name, 0.1);
        
        % for size calculation
        [~, ~, x, y] = extractSF_single(imds.Files{1}, power, 227);
        X = x;
        Y = zeros(length(imds.Files), length(y));
        parfor i = 1:length(imds.Files)
            [~, ~, ~, y] = extractSF_single(imds.Files{i}, power, 227); % x: x-xis (frequency), y: proportion of each freq component
            Y(i, :) = y; % all info
        end

    case {'officehome\Art', 'officehome\Clipart', 'officehome\Product', 'officehome\Real_World'}
        imds = extractImg(path0, name, 1);

        Y = [];
        for i = 1:length(imds.Files)
            [~, ~, x, y] = extractSF_single(imds.Files{i}, power, 227); % x: x-xis (frequency), y: proportion of each freq component
            Y = [Y; y]; % all info
        end
        X = x;

    case {'cifar-10', 'mnist'}
        %% variable setting
        prefix = varargin{1};
        ntot = varargin{2};
        nclass = varargin{3};

        %% filepath validity check
        if rem(ntot, nclass) ~= 0
            error("ntot should be multiple of nclass")
        end
        if nargin < 2
            ntot = 100;
            nclass = 10;
        end
        
        path = [path0 '\data\' name];
        if ~isfolder(path)
            error("Invalid path!")
        end
        
        %% extract file paths
        [filepath, ~] = extractFilepath_Label(path0, name, prefix, ntot, nclass, 'train');
        
        %% frequency distribution generation
        Y = [];
        for i = 1:ntot
            [~, ~, x, y] = extractSF_single(filepath{i}, power); % x: x-xis (frequency), y: proportion of each freq component
            Y = [Y; y]; % all info
        end
        X = x;

end

end