clc;
clear all;
close all;% iPhone models
models = {'iPhone 12', 'iPhone 13', 'iPhone 14', 'iPhone 15'};

% Random Forest predictions
rf_predictions = [2.8672, 3.0495, 3.2565, 3.4702];

% Monte Carlo predictions - Confidence Intervals (Lower and Upper bounds)
mc_lower_bound = [2.8318, 3.0368, 3.2453, 3.4621];
mc_upper_bound = [2.9027, 3.0622, 3.2677, 3.4784];

% Calculate the error bars (difference between upper/lower bounds and the Random Forest prediction)
mc_errors = [(rf_predictions - mc_lower_bound); (mc_upper_bound - rf_predictions)];

% Create the plot
figure;
hold on;

% Plot Random Forest predictions with error bars (Monte Carlo confidence intervals)
errorbar(1:4, rf_predictions, mc_errors(1, :), mc_errors(2, :), 'o-', 'LineWidth', 2);

% Set x-axis labels
xticks(1:4);
xticklabels(models);

% Label axes
xlabel('iPhone Model');
ylabel('Predicted Battery Life (Years)');

% Add title
title('Random Forest Predictions with Monte Carlo Confidence Intervals');

% Add grid
grid on;

hold off;
