function [] = visualizeFilter(net)
%% visualize conv1 filter
basicSettings;

Weights = net.Layers(2,1).Weights;
for channel = size(Weights, 3)
    for i = 1:96
        subplot(8, 12, i)
        filt = Weights(:, :, :, i);
        % filt = (filt - min(filt(:))) / (max(filt(:)) - min(filt(:)));

        imagesc(rgb2gray(rescale(filt)))
        colormap('gray')
        axis equal
        axis off
        % colormap(CustomColormap)
        % clim([-0.01 0.01])

    end
end

% 
% %%
% Weights = net2.Layers(2, 1).Weights 
% temp = [];
% for i = 1:96
%     temp(i) = mean(Weights(:), 'all');
% end
% histogram(Weights(:), 20)
% title('Example weight change 96 filters of conv1')
% xlabel('Weight change')
% ylabel('count')
% hold on
% w1 = net1.Layers(2, 1).Weights;
% histogram(w1(:))