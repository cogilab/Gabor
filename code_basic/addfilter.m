function [net] = addfilter(pathParam, net, suffix, varargin)
% initial weights & parameters
size_weights = size(net.Layers(2, 1).Weights);
Weights = zeros(size_weights);

% generate weights of conv1 filters
if contains(suffix, 'V1')
    corr = str2double(suffix(11:13)); % ex. _gabor_V1_0.3 -> 0.3
    seed = varargin{1};
    
    ablation = '';
    if contains(suffix, 'abLow')
        ablation = 'abLow';
    elseif contains(suffix, 'abMid')
        ablation = 'abMid';
    elseif contains(suffix, 'abHigh')
       ablation = 'abHigh';
    elseif contains(suffix, 'abnxny')
       ablation = 'abnxny';
    elseif contains(suffix, 'abSF')
       ablation = 'abSF';
    end

    [sf, ori, phase, nx, ny] = generate_gabor_param(size(Weights, 4), seed, corr, ablation);
    param_tot = [sf ori' phase' nx ny]';

    for channel = 1:size(Weights, 3)
        for ii = 1:size(Weights, 4)
            param = param_tot(:, ii)';

            filt = makeV1(size_weights(1:2), param);
            
            if contains(suffix, 'pxshuffle')
                filt = reshape(filt(randperm(length(filt(:)))), size_weights(1), size_weights(2));
            elseif contains(suffix, 'shuffle')
                filt = freq_shuffle(filt);
            end
            Weights(:, :, channel, ii) = filt;
        end
    end

elseif contains(suffix, 'gabor') || contains(suffix, 'comb') || contains(suffix, 'gau')
    param_tot = load([pathParam 'param' suffix '.mat']);
    for channel = 1:size(Weights, 3)
        for ii = 1:size(Weights, 4)
            param = param_tot.param_tot(:, ii)';

            filt = makeGabor(size_weights(1:2), param);
            if contains(suffix, 'gau')
                filt = filt / (2 * pi * param(1)^2);
                Weights(:, :, channel, ii) = filt;
            else
                filt = (filt - mean(filt(:))) / std(filt(:));
                Weights(:, :, channel, ii) = filt ...
                * sqrt(2 / (prod(size_weights(1:3))));
            end
        end
    end

elseif contains(suffix, 'dog')
    param_tot = load([pathParam 'param' suffix '.mat']);
    for channel = 1:size(Weights, 3)
        for ii = 1:size(Weights, 4)
            param = param_tot.param_tot(:, ii)';

            filt = makeDoG(size_weights(1:2), param);
            filt = (filt - mean(filt(:))) / std(filt(:));

            Weights(:, :, channel, ii) = filt ...
                .* sqrt(2 / (prod(size_weights(1:3))));
        end
    end
end

% new conv1 layers
conv1Layer = convolution2dLayer(size_weights(1:2), size_weights(4), 'Name', 'conv1', ...
    'NumChannels', size_weights(3), 'Stride', [4 4], 'Weights', Weights, 'Bias', []);

% network assemble
net = replaceLayer(net, 'conv1', conv1Layer);
net = dlnetwork(layerGraph(net), initialize=true);



%%% FUNCTIONS FOR V1 SAMPLING
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
            case 'abSF'
                sf_dist = [1, 1, 1, 1, 1, 1, 1, 1];
        
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
    ori(ori < 0) = ori(ori < 0) + pi;

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

    if ~isempty(varargin)
        if strcmp('abnxny', varargin{1})
            samps = mvnrnd([0, 0], cov_mat, features);
            samps_cdf = normcdf(samps);
            ny = interp1([0 cumsum(nx_dist)], log10(nx_bins), samps_cdf(:, 1), 'linear', 'extrap');
            ny = 10.^ny;
        end
    end

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

function outputImage = freq_shuffle(inputImage)
    F = fft2(inputImage);

    % Get the magnitude and phase
    magnitude = abs(F);
    phase_f = angle(F);
    
    % Shuffle the magnitudes
    shuffledMagnitude = magnitude(randperm(numel(magnitude)));
    shuffledMagnitude = reshape(shuffledMagnitude, size(magnitude));
    
    % Reconstruct the Fourier transform with shuffled magnitude and original phase
    F_shuffled = shuffledMagnitude .* exp(1i * phase_f);
    
    % Perform the inverse 2D Fourier Transform
    outputImage = ifft2(F_shuffled);
    
    % Since the output may have complex values, take the real part
    outputImage = real(outputImage);
end

end