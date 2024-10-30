function [filepath, labels] = extractFilepath_Label(path0, name, ntot, nclass, cond)
%% Exract Spatial Frequency of Images
% input: 
% path0 = project folder
% name = dataset name e.g. 'mnist'
% prefix = 'r' for resized, '' for original
% ntot = total # of images
% nclass= total # of classes
% cond = 'test' or 'train
% is_only_filepath = 1 if only filepath returning
% output: mean distribution of spatial frequency, filepath

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
filepath = cell(ntot, 1); ind = 1;
labels = zeros(ntot, 1);
switch name
    case 'imagenet'
        path1 = [path '\' cond '\'];
        class_list = dir(path1); class_list(1:2) = [];
        for class = 1:10
            path2 = [path1 class_list(class).name '\images\'];
            img_list = dir(path2); img_list(1:2) = [];
            for img = randsample(1:length(img_list), ntot/nclass)
                filepath{ind, 1} = [path2 img_list(img, 1).name];
                ind = ind+1;
            end
        end
        labels = reshape(repmat(1:10, ntot/nclass, 1), 1, [])';
    case 'mnist'
        path1 = [path '\' cond '\'];
        class_list = dir(path1); class_list(1:2) = []; 
        for class = 1:10
            path2 = [path1 class_list(class).name '\'];
            img_list = dir(path2); img_list(1:2) = [];
            for img = randsample(1:length(img_list), ntot/nclass)
                filepath{ind, 1} = [path2 img_list(img, 1).name];
                ind = ind+1;
            end
        end
        labels = reshape(repmat(1:10, ntot/nclass, 1), 1, [])';
    case 'cifar-10'
        path1 = [path '\' cond '\'];
        class_list = dir(path1); class_list(1:2) = []; 
        for class = 1:10
            path2 = [path1 class_list(class).name '\'];
            img_list = dir(path2); img_list(1:2) = [];
            for img = randsample(1:length(img_list), ntot/nclass)
                filepath{ind, 1} = [path2 img_list(img, 1).name];
                ind = ind+1;
            end
        end
        labels = reshape(repmat(1:10, ntot/nclass, 1), 1, [])';
    % case 'cifar-10'
    %     class_list = {'batch1', 'batch2','batch3','batch4','batch5'}; ind=1;
    %     for class = 1:nclass
    %         path1 = [path '\' class_list{class} '\'];
    %         data = load([path '\' sprintf('data_batch_%d.mat', class)]);
    %         labels(ind:ind + ntot/nclass - 1) = data.labels(1:ntot/nclass);
    %         for img = 1:ntot/nclass
    %             filepath{ind, 1} = [path1 prefix sprintf('%05d.png', img)];
    %             ind = ind+1;
    %         end
    % end
end

labels = categorical(labels);

end