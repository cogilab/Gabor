%%% Path Setting
path0 = 'your-project-path';
addpath([path0 '/code_basic'])

pathResult = [path0 '/results/'];
pathResult_analysis = [pathResult 'results_analysis/'];
pathResult_simul = [pathResult 'results_simulation/'];
pathResult_simul_save = pathResult_simul;

pathData = [path0 '/data/'];
pathData_img = [pathData 'images/'];

load([path0 '/data/palette_domain.mat'])
load([path0 '/data/palette_network.mat'])

%%% Figure setting
size_line = 0.5297;

%%% Defulat figure setting
set(0, 'DefaultAxesFontSize', 6);        % Default font size for axis labels
set(0, 'DefaultAxesLabelFontSizeMultiplier', 7/6);
set(0, 'DefaultLineLineWidth', size_line);        % Default line width for plots
set(0, 'DefaultAxesLineWidth', size_line);      % Default axes line width
set(0, 'DefaultAxesXColor', 'k');       % Set default x-axis color to black
set(0, 'DefaultAxesYColor', 'k');       % Set default y-axis color to black
set(0, 'DefaultTextColor', 'k');        % Set default text color to black (for titles and labels)


