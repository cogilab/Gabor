function [b] = mj_error_bar_dot(x, data, axis, varargin)
if axis == 2
    data = data';
end

m = mean(data, 1);
err = std(data, 1) / sqrt(size(data, 2));

b = bar(x, m);
hold on

er = errorbar(x, m, err);
er.Color = [1 0 0];
er.LineStyle = 'none';
er.CapSize = 0;
hold on

x_noise = normrnd(0,0.04,[1,numel(data)]);
s = scatter(repelem(1:size(data, 2),size(data,1)) + x_noise, reshape(data, [], 1));
s.MarkerEdgeColor = 'none';
s.MarkerFaceColor = [0.3 0.3 0.3];
s.MarkerFaceAlpha = 0.3; 
s.SizeData = 10;

hold off

end