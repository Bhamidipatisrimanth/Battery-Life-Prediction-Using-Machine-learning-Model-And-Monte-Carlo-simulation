clc;
clear all;
close all;

% Specify the data file
dataFile = 'iPhone_15_New.csv';

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
numSimulations = 2000;
monteCarloPredictions = monteCarloSimulation(randomForestModel, X, numSimulations);

% Compare the distribution of Random Forest and Monte Carlo predictions
figure;
subplot(2, 1, 1);
histogram(YPred_RF, 30);
xlabel('Predicted Battery Life (Years)');
ylabel('Frequency');
title('Random Forest Model Predictions on Test Set');

subplot(2, 1, 2);
histogram(monteCarloPredictions, 30);
xlabel('Predicted Battery Life (Years)');
ylabel('Frequency');
title('Monte Carlo Simulation Predictions');

%% New User Input Prediction

% Define new user input for prediction
newUserInput = [5, 7, 2, 3, 1, 25, 3349];  % Example input: [Screen_Time_Hours, Charging_Cycles_Per_Week, Bluetooth_Hours, Cellular_Data_Hours, GPS_Hours, Ambient_Temperature_Celsius, Battery_Capacity_mAh]

% Predict battery life for new user input using the Random Forest model
predictedLife = predict(randomForestModel, newUserInput);

% Display the predicted battery life for the new user
disp(['Predicted Battery Life for New User Input: ', num2str(predictedLife), ' years']);

% Incorporate variability from Monte Carlo simulation
% Calculate variability factors (standard deviation) from Monte Carlo predictions
std_MC = std(monteCarloPredictions);
mean_MC = mean(monteCarloPredictions);

% Adjust the prediction for new user input by adding/subtracting the variability
adjustedPredictionLower = predictedLife - std_MC;
adjustedPredictionUpper = predictedLife + std_MC;

% Display adjusted prediction range
disp(['Adjusted Predicted Battery Life Range (incorporating variability using Montecarlo): ', num2str(adjustedPredictionLower), ' - ', num2str(adjustedPredictionUpper), ' years']);
