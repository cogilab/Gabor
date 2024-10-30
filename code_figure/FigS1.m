%% Desciprions
% This script is to reproduce figures from supplementary Fig 1. 

%% Fig. S1b
clear; close; clc;
path0 = '/home/dgxadmin/Minjun/Project-DNN_Gabor';
addpath([path0 '/code_basic'])
basicSettings;

[sf, ori, phase, nx, ny] = generate_gabor_param(96, 2, 0.8, '');
param_tot = [sf ori' phase' nx ny]';

[~, inds] = sort(ori, 'ascend');
param_tot = param_tot(:, inds);


% 1. orienation
f = figure; f.Position = [500 500 200 200];
x = [-22.5, 22.5, 67.5, 112.5, 157.5] * pi / 180;
histogram(ori, x); hold on

y = [66, 49, 77, 54] * 96 / sum([66, 49, 77, 54]);
plot(x(1:end-1) + pi/8, y)
xticks(0:pi/4:pi)
set(gca,'TickDir','out');
xlabel('orientation');
ylabel('count')
yticks(0:10:30);

% 2. sf
f = figure; f.Position = [500 500 200 200];
bins = [0.5 0.7 1.0 1.4 2.0 2.8 4 5.6 8.0 11.2];
histogram(sf, bins); hold on
x = [0.5916    0.8367    1.1832    1.6733    2.3664    3.3466    4.7329    6.6933];
y = [4,  4,  8, 25, 32, 26, 28, 12] * 96 / sum([4,  4,  8, 25, 32, 26, 28, 12]);
plot(x, y)
set(gca, 'xscale', 'log')
box off
xticks(bins)
set(gca,'TickDir','out');
xlabel('spatial frequency (cpd)');
ylabel('count')

% 3. nx, ny
f = figure; f.Position = [500 500 200 200];
scatter(nx, ny, '.')
xlim([0 1]); ylim([0 1.5])
xticks(0:0.5:1); yticks(0:0.5:1.5)
set(gca,'TickDir','out');
xlabel('nx');
ylabel('ny');


%% Fig. S1d
figure;
Weights = zeros(30, 30, 3, 96);

for ii = 1:size(Weights, 4)
    param = param_tot(:, ii)';

    filt = makeV1([30 30], param);
    for jj = 1:3
        Weights(:, :, jj, ii) = rescale(filt) - mean(rescale(filt(:))) + 0.5;
    end
end
montage(Weights(:, :, :), 'BorderSize', [3 3], 'BackgroundColor', [0 0  0], 'Size', [8 12]);


%% Subfunctions
function [sf, ori, phase, nx, ny] = generate_gabor_param(features, seed, sf_corr, varargin)
    rng(seed);
    
    % phase - random sampling
    phase_bins = [0, 360] * pi / 180;
    phase_dist = [1];
    sf_max = 9;
    sf_min = 0.1;
    
    % orientation - DeValois 1982
    ori_bins = [-22.5, 22.5, 67.5, 112.5, 157.5] * pi / 180;
    ori_dist = [66, 49, 77, 54];
    ori_dist = ori_dist / sum(ori_dist);
    
    % covariance of sf and nx - Schiller 1976
    cov_mat = [1, sf_corr; sf_corr, 1];
    
    % nx, ny - Ringach 2002
    nx_bins = logspace(-1, 0.2, 6); nx_bins(end) = [];
    ny_bins = logspace(-1, 0.2, 6);
    n_joint_dist = [2.,  0.,  1.,  0.,  0.;
                    8.,  9.,  4.,  1.,  0.;
                    1.,  2., 19., 17.,  3.;
                    0.,  0.,  1.,  7.,  4.];
    n_joint_dist(n_joint_dist == 0) = 1e-10;
    n_joint_dist = n_joint_dist / sum(n_joint_dist(:));
    nx_dist = sum(n_joint_dist, 2)';
    nx_dist = nx_dist / sum(nx_dist);
    ny_dist_marg = n_joint_dist ./ sum(n_joint_dist, 2);
    
    % sf - DeValois 1982
    sf_bins = [0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8];
    sf_dist = [4,  4,  8, 25, 32, 26, 28, 12];

    if ~isempty(varargin)
        switch varargin{1}
            case 'abLow'
                sf_dist = [1e-10,  1e-10,  1e-10,  1e-10, 32, 26, 28, 12];
            case 'abMid'
                sf_dist = [4,  4,  8, 25, 1e-10, 1e-10, 28, 12];
            case 'abHigh'
                sf_dist = [4,  4,  8, 25, 32, 26, 1e-10, 1e-10];
        end
    end
   
    sfmax_ind = find(sf_bins <= sf_max, 1, 'last');
    sfmin_ind = find(sf_bins >= sf_min, 1, 'first');

    sf_bins = sf_bins(sfmin_ind:sfmax_ind);
    sf_dist = sf_dist(sfmin_ind:sfmax_ind-1);

    sf_dist = sf_dist / sum(sf_dist);

    % samplings
    phase = sample_dist(phase_dist, phase_bins, features);
    ori = sample_dist(ori_dist, ori_bins, features);
    % ori(ori < 0) = ori(ori < 0) + pi;

    samps = mvnrnd([0, 0], cov_mat, features);
    samps_cdf = normcdf(samps);

    nx = interp1([0 cumsum(nx_dist)], log10(nx_bins), samps_cdf(:, 1), 'linear', 'extrap');
    nx = 10.^nx;

    ny_samp = rand(features, 1);
    ny = zeros(features, 1);
    for samp_ind = 1:features
        bin_id = find(nx_bins < nx(samp_ind), 1, 'last');
        ny(samp_ind) = interp1([0 cumsum(ny_dist_marg(bin_id, :))], log10(ny_bins), ny_samp(samp_ind), 'linear', 'extrap');
    end
    ny = 10.^ny;

    sf = interp1([0 cumsum(sf_dist)], log2(sf_bins), samps_cdf(:, 2), 'linear', 'extrap');
    sf = 2.^sf;
end

function rand_sample = sample_dist(hist, bins, ns, varargin)
    rand_sample = rand(1, ns);
    if ~isempty(varargin)
        scale = varargin{1};
    else
        scale = 'linear';
    end
    switch scale
        case 'linear'
            rand_sample = interp1([0, cumsum(hist)], bins, rand_sample, 'linear', 'extrap');
        case 'log2'
            rand_sample = interp1([0, cumsum(hist)], log2(bins), rand_sample, 'linear', 'extrap');
            rand_sample = 2.^rand_sample;
        case 'log10'
            rand_sample = interp1([0, cumsum(hist)], log10(bins), rand_sample, 'linear', 'extrap');
            rand_sample = 10.^rand_sample;
        otherwise
            error('Invalid scale option. Use ''linear'', ''log2'', or ''log10''.');
    end
end