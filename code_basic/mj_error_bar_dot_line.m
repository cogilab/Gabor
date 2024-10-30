function [b] = mj_error_bar_dot_line(x, data, colors, axis)
if axis == 2
    data = data'; % data should be n x conditions
end

m = mean(data, 1);
err = std(data, 1) ./ sqrt(size(data, 1));

b = bar(x, m);
b.FaceColor = 'flat';
b.CData = colors;
hold on

er = errorbar(x, m, err);
er.Color = [0 0 0];
er.LineStyle = 'none';
er.CapSize = 0;
hold on

color = [.3 .3 .3];
alpha = [.7];

s = scatter(repelem(1:size(data, 2),size(data,1)), reshape(data, [], 1));
s.MarkerEdgeColor ='none';
s.MarkerFaceColor = color;
s.MarkerFaceAlpha = alpha; s.MarkerEdgeAlpha = alpha;
s.SizeData = 10;
% s.AlphaData = ones(length(reshape(data, [], 1)), 1) * 10;
% s.MarkerFaceAlpha = 'flat';

for nn = 1:size(data, 1)
    plot(1:size(data, 2), data(nn, :), 'Color', [color alpha])
end

hold off

end