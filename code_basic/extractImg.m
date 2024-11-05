function [imds] = extractImg(pathData_img, name, varargin)
%% Extract images from the pathData_img
% path validity check
path = [pathData_img name];
if ~isfolder(path)
    disp(path)
    error("Invalid path!")
end

% image extraction
imds = imageDatastore(path,"FileExtensions",[".jpg", ".png", ".JPEG"], "IncludeSubfolders",true, ...
    "LabelSource","foldernames");
labels = zeros(length(imds.Labels), 1);
labels_name = string(unique(imds.Labels));

for ii = 1:length(labels_name)
    labels(string(imds.Labels) == labels_name(ii)) = ii;
end

imds.Labels = labels;

% (optional) only use subset of images
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

end