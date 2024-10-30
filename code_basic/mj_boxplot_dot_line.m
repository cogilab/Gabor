function [] = mj_boxplot_dot_line(data, boxColor, varargin)
if ~isempty(varargin)
    Widths = varargin{1};
else
    Widths = 0.5;
end



h = boxplot(data, 'Colors', 'k', 'Widths', Widths); 
boxes = findobj(gca, 'Tag', 'Box');
for k = 1:length(boxes)
    % Set the edge color of the box to none (remove the stroke)
    set(boxes(k), 'Color', 'none');
end

for j=1:length(boxes)
    patch(get(boxes(j),'XData'),get(boxes(j),'YData'), boxColor(length(boxes)+1-j, :), 'EdgeColor', 'none');
end

hold on;
medians = findobj(gca, 'Tag', 'Median');
for k = 1:length(medians)
    medianData = get(medians(k), 'YData');
    medianX = get(medians(k), 'XData');
    plot(medianX, medianData, 'Color', [1 1 1], 'LineStyle', '-');
end

% Set properties for the whiskers
whiskers = findobj(gca, 'Tag', 'Lower Whisker');
set(whiskers, 'Color', 'k', 'LineStyle', '-');
whiskers = findobj(gca, 'Tag', 'Upper Whisker');
set(whiskers, 'Color', 'k', 'LineStyle', '-');

% Scatter plot
shift = 0.3;
color = [.3 .3 .3];
alpha = [.3];
s = scatter(repelem(1:size(data, 2),size(data,1)) + shift, reshape(data, [], 1));
s.MarkerEdgeColor ='none';
s.MarkerFaceColor = color;
s.MarkerFaceAlpha = alpha; s.MarkerEdgeAlpha = alpha;
s.SizeData = 10;
% s.AlphaData = ones(length(reshape(data, [], 1)), 1) * 10;
% s.MarkerFaceAlpha = 'flat';

for nn = 1:size(data, 1)
    plot((1:size(data, 2)) +shift, data(nn, :), 'Color', [color alpha])
end

hold off