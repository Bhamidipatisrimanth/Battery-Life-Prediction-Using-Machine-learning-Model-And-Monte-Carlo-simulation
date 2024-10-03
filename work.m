clc;
clear all;
close all;

% Load the data
data = readtable('iPhone_12_new.csv');

% Prepare the input features and target variable
% Using the column names you provided
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
% 100 trees are commonly used; you can adjust this based on your needs
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

% Predict battery life on the test set
YPred = predict(randomForestModel, XTest);

% Calculate Mean Squared Error on the test set
mseTest = mean((YPred - YTest).^2);
disp(['Test Set Mean Squared Error: ', num2str(mseTest)]);

% Plot the feature importance
figure;
bar(randomForestModel.OOBPermutedPredictorDeltaError);
xlabel('Feature Index');
ylabel('Out-of-Bag Feature Importance');
xticklabels({'Screen Time', 'Charging Cycles', 'Bluetooth Hours', 'Cellular Data Hours', 'GPS Hours', 'Ambient Temp', 'Battery Capacity'});
title('Feature Importance Estimates');
