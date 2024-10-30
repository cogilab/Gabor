function [imds] = extractImg(pathData_img, name, varargin)
%% Description 
%%% Function Objective %%%
% To extract a dataset named "name" as a single structure of data/label
%%% Input %%%
% path0: path of project foler
% name: 'cifar-10' or 'mnist'
% numTot: total number of images
% numClass: total number of classes
% option: 'train', 'valid', 'test'
% option2: 'rgb', 'gray'
%%% Output %%%
% data = structure type
% data.input = w x h x c x numTot array
% data.output = numTot x 1 array (classes)
% switch name
%     case {'pacs\photo', 'pacs\art_painting', 'pacs\cartoon', 'pacs\sketch', ...
%             'officehome\Art', 'officehome\Clipart', 'officehome\Product', 'officehome\Real_World', ...
%             'imagenet-sketch', 'cifar-10\train'}
%% class indxs
path = [pathData_img name];
if ~isfolder(path)
    disp(path)
    error("Invalid path!")
    
end

imds = imageDatastore(path,"FileExtensions",[".jpg", ".png", ".JPEG"], "IncludeSubfolders",true, ...
    "LabelSource","foldernames");
labels = zeros(length(imds.Labels), 1);
labels_name = string(unique(imds.Labels));

for ii = 1:length(labels_name)
    labels(string(imds.Labels) == labels_name(ii)) = ii;
end

imds.Labels = labels;

% class for manual selection
if ~isempty(varargin)
    p = varargin{1};
    ind_list = ones(1, length(imds.Files));
    for class = 1:length(labels_name)
        ind1 = find(imds.Labels == class, 1);
        if class ~= length(labels_name)
            ind2 = find(imds.Labels == (class+1), 1);
        else
            ind2 = length(imds.Files)+1;
        end

        len = floor((ind2 - ind1) * p);
        ind_selected = randperm(ind2-ind1, len) + ind1 - 1;

        ind_list(ind_selected) = 0; % selected indices
    end
    
    imds.Files(logical(ind_list)) = [];           
end

%% display extracted image profiles
disp(['Data name = ' name])
fprintf('# total img = %d\n', length(imds.Labels))
fprintf("# classes = %d\n", length(labels_name))
str = '# images = (';

for ii = 1:length(labels_name)
    str = [str num2str(length(find(imds.Labels == ii))) '/'];
end
str(end) = []; str = [str ')'];
disp(str)
fprintf("\n")


%     case {'mnist'}
%         %% variable setting
%         numTot = varargin{1};
%         numClass = varargin{2};
%         option = varargin{3};
% 
%         %% class indxs
%         class_indx = 1:numClass;
% 
%         %% filepath validity check
%         if rem(numTot, numClass) ~= 0
%             error("numTot should be multiple of numClass")
%         end
% 
%         path = [path0 '\data\' name];
%         if ~isfolder(path)
%             error("Invalid path!")
%         end
% 
%         %% extract file paths & labels
%         filepath = cell(numTot, 1); 
%         labels = zeros(numTot, 1);
%         ind = 1;
% 
%         % class lists
%         path1 = [path '\' option '\'];
%         class_list = dir(path1); class_list(1:2) = []; 
% 
%         % extraction
%         for class = class_indx
%             path2 = [path1 class_list(class).name '\'];
%             img_list = dir(path2); img_list(1:2) = [];
% 
%             img_select_mode = 'order'; %'order';
%             switch img_select_mode
%                 case 'rand'
%                     img_indx = randsample(1:length(img_list), numTot/numClass);
%                 case 'order'
%                     img_indx = 1:(numTot/numClass);
%             end
%             for img = img_indx
%                 filepath{ind, 1} = [path2 img_list(img, 1).name];
%                 labels(ind, 1) = class;
%                 ind = ind+1;
%             end
%         end
% 
%         %% read images using image data store
%         imds = imageDatastore(filepath);
%         imds.Labels = labels;
% 
% end

end