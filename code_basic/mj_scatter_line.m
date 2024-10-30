function [rsq, p] = mj_scatter_line(a, b, varargin)
    assert(length(a) == length(b), 'Two inputs should have same length')
    assert(isvector(a) && isvector(b), 'Two inptus should be vector')
    
    if size(a, 2) ~= 1
        a = a';
    end
    if size(b, 2) ~= 1
        b = b';
    end
    
    if ~isempty(varargin)
        Color = varargin{1};
    else
        Color = [.5 .5 .5];
    end

    scatter(a,b, 20, 'filled', 'MarkerFaceColor', Color)
    hold on

    mdl = fitlm(a, b);
    
    % Add regression line to plot
    plot(a,mdl.Fitted, 'Color', Color, 'LineWidth', 3);

    % p and r
    rsq = mdl.Rsquared.Adjusted;
    p = table2array(mdl.Coefficients(2, 4));
end