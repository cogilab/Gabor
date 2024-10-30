function [blob] = makeBlob(size, params, varargin)
%% Make Gabor Filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. size : size of gabor filters e.g. [11 11]
% 2. params : [sigma, gamma, psi]
% 3. dog (output): z-normalized difference of gaussians (DoG) filters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % parameter - examples
% size = [29 29];
% sigma = 3; 
% psi = 1;

%% parameter extraction
sigma=params(1); psi=params(2); 

% Gabor filter generation
[x, y] = meshgrid(-(size(2)-1)/2:1:(size(2)-1)/2, (size(1)-1)/2:-1:-(size(1)-1)/2);
blob = psi * exp(-(x.^2 + y.^2)/(2*sigma^2)) / (2*pi*sigma^2);
blob = rescale(blob);
blob = blob - blob(end, end);

% for high pass blobs
if ~isempty(varargin)
    if strcmp(varargin{1}, 'high')
        blob = max(blob(:)) - blob;
    end
end
