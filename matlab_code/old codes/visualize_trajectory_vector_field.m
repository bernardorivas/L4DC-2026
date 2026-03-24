% visualize_trajectory_vector_field.m
% Visualizes trajectory data with vector field arrows
%
% Data structure in .mat files:
%   - trajectory_data: all trajectory points
%   - F_field: vector field f(x) evaluated at trajectory_data

clear; close all; clc;

file1 = '/Users/bdoprad/Work/Code/Active/ACC2026/examples/example_1_tau_1_0/toggle_switch_data.mat';
file2 = '/Users/bdoprad/Work/Code/Active/ACC2026/examples/example_2_tau_1_0/piecewise_vdp_data.mat';

if ~exist(file1, 'file')
    error('Data file not found: %s', file1);
end
if ~exist(file2, 'file')
    error('Data file not found: %s', file2);
end

data1 = load(file1);
data2 = load(file2);

X1 = data1.trajectory_data;
Y1 = data1.F_field;
X2 = data2.trajectory_data;
Y2 = data2.F_field;

bounds1 = [0, 6; 0, 6];
bounds2 = [-3, 3; -3, 3];

figure('Position', [100, 100, 1400, 600]);

subplot(1, 2, 1);
plot_trajectory_vector_field(X1, Y1, bounds1, 'Toggle Switch');

subplot(1, 2, 2);
plot_trajectory_vector_field(X2, Y2, bounds2, 'Piecewise Van der Pol');

saveas(gcf, 'trajectory_vector_field_visualization.png');

function plot_trajectory_vector_field(X, Y, bounds, title_str)
    hold on;
    scatter(X(:,1), X(:,2), 15, 'y*', 'MarkerFaceAlpha', 0.4, ...
            'DisplayName', 'Trajectory points');
    quiver(X(:,1), X(:,2), Y(:,1), Y(:,2), ...
           'b', 'LineWidth', 1.0, 'MaxHeadSize', 0.5, 'AutoScale', 'on', ...
           'AutoScaleFactor', 1.0, 'DisplayName', 'Vector field f(x)');
    hold off;

    xlim(bounds(1, :));
    ylim(bounds(2, :));
    xlabel('x_1');
    ylabel('x_2');
    title(title_str);
    grid on;
    axis equal;
    box on;
    legend('Location', 'best');

    text(0.02, 0.98, sprintf('%d points', size(X, 1)), ...
         'Units', 'normalized', 'VerticalAlignment', 'top', ...
         'FontSize', 9, 'BackgroundColor', 'white');
end
