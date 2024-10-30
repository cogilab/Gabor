function [b] = mj_error_bar(x, data, axis, varargin)
if axis == 2
    data = data';
end

m = mean(data, 1);
err = std(data, 1);

b = bar(x, m);


hold on
er = errorbar(x, m, err);
er.Color = [0 0 0];
er.LineStyle = 'none';
er.CapSize = 0;
hold off

end