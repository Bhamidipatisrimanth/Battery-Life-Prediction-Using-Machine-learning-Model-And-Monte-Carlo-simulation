clc;
clear all;
close all;

% Load the data
data = readtable('iPhone_12_new.csv');

% Prepare the input features and target variable
features = data(:, {'Screen_Time_Hours', 'Charging_Cycles_Per_Week', 'Bluetooth_Hours', ...
                    'Cellular_Data_Hours', 'GPS_Hours', 'Ambient_Temperature_Celsius', ...
                    'Battery_Capacity_mAh'});
target = data.Estimated_Lifespan_Years;

% Convert the table to arrays for machine learning
X = table2array(features);
Y = target;  % No need to use table2array since 'target' is a single column vector

% Split the data into training (70%) and testing (30%) sets
cv = cvpartition(size(X,1), 'HoldOut', 0.3);
XTrain = X(training(cv), :);
YTrain = Y(training(cv));
XTest = X(test(cv), :);
YTest = Y(test(cv));

% Train the Random Forest model
numTrees = 100;
randomForestModel = TreeBagger(numTrees, XTrain, YTrain, 'Method', 'regression', ...
                               'OOBPrediction', 'On', 'OOBPredictorImportance', 'On');

% Evaluate the model using Out-of-Bag Mean Squared Error
oobError = oobError(randomForestModel);
figure;
plot(oobError);
xlabel('Number of Grown Trees');
ylabel('Out-of-Bag Mean Squared Error');
title('Out-of-Bag Mean Squared Error for Random Forest');

% Predict battery life on the test set using Random Forest model
YPred_RF = predict(randomForestModel, XTest);

% Calculate Mean Squared Error on the test set for Random Forest model
mseTest_RF = mean((YPred_RF - YTest).^2);
disp(['Test Set Mean Squared Error (Random Forest): ', num2str(mseTest_RF)]);

% Plot the feature importance
figure;
bar(randomForestModel.OOBPermutedPredictorDeltaError);
xlabel('Feature Index');
ylabel('Out-of-Bag Feature Importance');
xticklabels({'Screen Time', 'Charging Cycles', 'Bluetooth Hours', 'Cellular Data Hours', 'GPS Hours', 'Ambient Temp', 'Battery Capacity'});
title('Feature Importance Estimates');

%% Monte Carlo Simulation

% Number of Monte Carlo simulations
numSimulations = 1000;

% Define distributions for each input feature (example: normal distribution based on mean and std)
% These values should be adjusted based on your data characteristics
screenTime_mean = mean(X(:,1));
screenTime_std = std(X(:,1));
chargingCycles_mean = mean(X(:,2));
chargingCycles_std = std(X(:,2));
bluetoothHours_mean = mean(X(:,3));
bluetoothHours_std = std(X(:,3));
cellularDataHours_mean = mean(X(:,4));
cellularDataHours_std = std(X(:,4));
gpsHours_mean = mean(X(:,5));
gpsHours_std = std(X(:,5));
ambientTemp_mean = mean(X(:,6));
ambientTemp_std = std(X(:,6));
batteryCapacity_mean = mean(X(:,7));
batteryCapacity_std = std(X(:,7));

% Generate random samples for each feature based on normal distribution
randScreenTime = normrnd(screenTime_mean, screenTime_std, numSimulations, 1);
randChargingCycles = normrnd(chargingCycles_mean, chargingCycles_std, numSimulations, 1);
randBluetoothHours = normrnd(bluetoothHours_mean, bluetoothHours_std, numSimulations, 1);
randCellularDataHours = normrnd(cellularDataHours_mean, cellularDataHours_std, numSimulations, 1);
randGpsHours = normrnd(gpsHours_mean, gpsHours_std, numSimulations, 1);
randAmbientTemp = normrnd(ambientTemp_mean, ambientTemp_std, numSimulations, 1);
randBatteryCapacity = normrnd(batteryCapacity_mean, batteryCapacity_std, numSimulations, 1);

% Combine all random samples into a matrix
monteCarloInput = [randScreenTime, randChargingCycles, randBluetoothHours, ...
                   randCellularDataHours, randGpsHours, randAmbientTemp, ...
                   randBatteryCapacity];

% Predict battery life for each Monte Carlo simulation
monteCarloPredictions = predict(randomForestModel, monteCarloInput);

% Analyze the results: plot histogram of the predicted battery life
figure;
histogram(monteCarloPredictions, 30);
xlabel('Predicted Battery Life (Years)');
ylabel('Frequency');
title('Monte Carlo Simulation of Predicted Battery Life');

% Calculate 95% Confidence Interval for Monte Carlo predictions
CI_MonteCarlo = prctile(monteCarloPredictions, [2.5 97.5]);
disp(['95% Confidence Interval for Predicted Battery Life (Monte Carlo): ', num2str(CI_MonteCarlo(1)), ' - ', num2str(CI_MonteCarlo(2))]);

% Calculate 95% Confidence Interval for Monte Carlo predictions
CI_YTest = prctile(YPred_RF, [2.5 97.5]);
disp(['95% Confidence Interval for Predicted Battery Life (RandomForestTest): ', num2str(CI_YTest(1)), ' - ', num2str(CI_YTest(2))]);


%% Compare Random Forest Model Predictions with Monte Carlo Simulation


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
