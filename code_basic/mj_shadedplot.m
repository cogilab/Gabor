function [] = mj_shadedplot(x, y, varargin)

if size(x, 2) == 1
    x = x';
elseif ~ismember(1, size(x))
    error('x must be vector')
end

if size(y, 1) == size(x, 2)
    y = y';
elseif ~ismember(size(x, 2), size(y))
    error('x and y must have same length')
end

% color = [.5 .5 .5];
linewidth = 1.5;

if length(varargin) == 1
    color = varargin{1};
elseif length(varargin) == 2
    color = varargin{1};
    version = 'err';
elseif length(varargin) > 2
    error("Undefined Varargin - mj_shadedplot")
end

N = size(y, 1);
if N == 1
   plot(x, y, 'Color', color, 'LineWidth', linewidth); 
elseif N > 1
   y_mean = mean(y);
   y_err = std(y);
   hold on
   plot(x, y_mean, 'Color', color, 'LineWidth', linewidth);  
   patch([x flip(x)], [y_mean-y_err flip(y_mean+y_err)], color, ...
       'FaceAlpha',0.3, 'EdgeColor','none'); 
   hold off

end


end