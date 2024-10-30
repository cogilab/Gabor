function [gb] = makeGabor(size, params)
%% Make Gabor Filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. size : size of gabor filters e.g. [11 11]
% 2. params : [sigma, gamma, psi, theta, lambda];
% 3. gb (output): z-normalized gabor filters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% parameter - examples
% % %% parameter - examples
% size = 11;
% sigma = 1; 
% gamma = 0.5;
% psi = pi/2;
% theta = pi;
% b = 1;

%% parameter extraction
sigma=params(1); gamma=params(2); psi=params(3); theta=params(4); lambda=params(5);

% Gabor filter generation
[x, y] = meshgrid(1-(1+size/2):1:size-(1+size)/2, size-(1+size)/2:-1:1-(1+size/2));
x_theta = x*cos(theta) + y*sin(theta);
y_theta = -x*sin(theta) + y*cos(theta);
sigma_x = sigma; 
sigma_y = sigma/gamma;
gb=exp(-0.5*(x_theta.^2/sigma_x^2+y_theta.^2/sigma_y^2)).*cos(2*pi/lambda*x_theta+psi);

