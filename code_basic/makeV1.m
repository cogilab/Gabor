function [gb] = makeV1(size, params)
%% Make Gabor Filter 
% [1 2 3 4 5] = [sf, ori, phase, nx, ny]

%% parameter extraction
sf=params(1) / (227/15); ori=params(2); phase=params(3); nx=params(4); ny=params(5);

% Gabor filter generation
[x, y] = meshgrid(1-(1+size/2):1:size-(1+size)/2, size-(1+size)/2:-1:1-(1+size/2));
x_theta = x*cos(ori) + y*sin(ori);
y_theta = -x*sin(ori) + y*cos(ori);
sigma_x = nx / sf; 
sigma_y = ny / sf;

gb=exp(-0.5*(x_theta.^2/sigma_x^2+y_theta.^2/sigma_y^2)).*cos(2*pi*sf*x_theta+phase) ./ (2*pi*sigma_x*sigma_y);

