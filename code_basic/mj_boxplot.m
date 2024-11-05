function [] = mj_boxplot(data, boxColor, varargin)
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
hold off;

% Set properties for the whiskers
whiskers = findobj(gca, 'Tag', 'Lower Whisker');
set(whiskers, 'Color', 'k', 'LineStyle', '-');
whiskers = findobj(gca, 'Tag', 'Upper Whisker');
set(whiskers, 'Color', 'k', 'LineStyle', '-');
