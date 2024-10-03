clc;
clear all;
close all;

% Specify the data file
dataFile = 'iPhone_12_new.csv';

% Train and test Random Forest model
[randomForestModel, YPred_RF, mseTest_RF] = trainAndTestRandomForest(dataFile);

% Display confidence interval for Random Forest predictions on test set
CI_YTest = prctile(YPred_RF, [2.5 97.5]);
disp(['95% Confidence Interval for Predicted Battery Life (RandomForestTest): ', num2str(CI_YTest(1)), ' - ', num2str(CI_YTest(2))]);

% Prepare data for Monte Carlo simulation
data = readtable(dataFile);
features = data(:, {'Screen_Time_Hours', 'Charging_Cycles_Per_Week', 'Bluetooth_Hours', ...
                    'Cellular_Data_Hours', 'GPS_Hours', 'Ambient_Temperature_Celsius', ...
                    'Battery_Capacity_mAh'});
X = table2array(features);

% Perform Monte Carlo simulation
numSimulations = 1000;
monteCarloPredictions = monteCarloSimulation(randomForestModel, X, numSimulations);

% Compare the distribution of Random Forest and Monte Carlo predictions
figure;
histogram(YPred_RF, 30);
hold on;
xline(CI_YTest(1), 'r--', 'LineWidth', 2);  % Lower bound of 95% CI for Monte Carlo
xline(CI_YTest(2), 'r--', 'LineWidth', 2);  % Upper bound of 95% CI for Monte Carlo
xlabel('Predicted Battery Life (Years)');
ylabel('Frequency');
title('Random Forest Model Predictions on Test Set');

