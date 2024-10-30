function [img, freq, X, Y] = extractSF_single(filepath, varargin)
%% Exract Spatial Frequency of Images
% input: filepath of a single image
% output: mean distribution of spatial frequency
%% Note
% from image, the period (#pixels) should range from 2 to length of
% hypotenuse
% 
%% image processing
% read image
img = imread(filepath);
img = im2double(img);

% rbg2gray & rescale
if size(img, 3) == 3
    img = rgb2gray(img); % convert unit8 -> double -> gray scale
end
img = rescale(img);

% % crop if needed
[h, w] = size(img);
if length(varargin) > 1
    imgsize = varargin{2};
    img = imresize(img, [imgsize imgsize]);
    h = imgsize; w = imgsize;
end

% normalize
img2 = img - mean(img, 'all');

% calculate frequency
freq = abs(fftshift(fft2(img2)));
if ~isempty(varargin)
    power = varargin{1};
    if power
        freq = freq .* freq;
    end
end

%% frequency distribution generation - xaxis
mapFreq = zeros(h, w);
center = h/2 + 1/2;
for i = 1:h
    for j = 1:w
        xloc = abs(center - i); yloc = abs(center - j);
        mapFreq(i, j) = sqrt((xloc / w)^2 + (yloc / h)^2);
    end
end
% mapFreq = mapFreq;

%% image distribution
[mapFreq_sort, order] = sort(reshape(mapFreq, 1, []));
freq_sort = reshape(freq, [], 1); freq_sort = freq_sort(order);
X = unique(mapFreq_sort);
Y = zeros(1, length(X));

for i = 1:length(X)
    Y(i) = mean(freq_sort(find(X(i) == mapFreq_sort)));
end

end