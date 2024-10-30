function [dog] = makeDoG(size, params)
%% Make Gabor Filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. size : size of gabor filters e.g. [11 11]
% 2. params : [sigma, gamma, psi]
% 3. dog (output): z-normalized difference of gaussians (DoG) filters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % parameter - examples
% size = [29 29];
% sigma = 3; 
% gamma = 2;
% psi = 1;

%% parameter extraction
sigma=params(1); gamma=params(2); psi=params(3); 

% Gabor filter generation
[x, y] = meshgrid(-(size(2)-1)/2:1:(size(2)-1)/2, (size(1)-1)/2:-1:-(size(1)-1)/2);
temp1 = exp(-(x.^2 + y.^2)/(2*sigma^2)) / (2*pi*sigma^2);
temp2 = exp(-(x.^2 + y.^2)/(2 * gamma^2 * sigma^2)) / (2 * pi * gamma^2 * sigma^2);
dog = psi * (temp1 - temp2);
dog = rescale(dog);

dog = (dog - mean(dog(:))) / std(dog(:));
dog = dog - dog(end, end);