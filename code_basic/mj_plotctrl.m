function [] = mj_plotctrl(size)
% Set the units of the figure to centimeters
fig = gcf;
fig.Units = 'centimeters';

% Adjust the figure size to include the entire axes with labels and titles
fig.Position(1) = 30;
fig.Position(2) = 30;
fig.Position(3) = size(1)/10 + 2;
fig.Position(4) = size(2)/10 + 2;

% Create axes in the figure
ax = gca;
ax.XColor = 'k';  % Set x-axis color to black
ax.YColor = 'k';  % Set y-axis color to black

ax.Title.Color = 'k';
ax.XLabel.Color = 'k';
ax.YLabel.Color = 'k';

% Set the units of the axes to centimeters
ax.Units = 'centimeters';

% Define the desired size of the plot area (30mm x 30mm = 3cm x 3cm)
desiredPlotWidth = size(1) /10; % in centimeters
desiredPlotHeight = size(2) /10; % in centimeters

% Calculate the position of the axes to fit the desired plot size
ax.Position = [1, 1, desiredPlotWidth, desiredPlotHeight];

axLength = max(ax.Position(3), ax.Position(4)); % minimum of width and height in cm
sz = 0.05 / axLength; % 0.5 cm is 5 mm

set(gca, 'tickdir', 'out', 'tickLength', [sz sz], 'Box', 'off');
