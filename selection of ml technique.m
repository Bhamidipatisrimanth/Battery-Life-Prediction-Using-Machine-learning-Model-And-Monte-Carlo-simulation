clc;
clear all;
close all;% Load the data
data = readtable('iPhone_12_new.csv');
disp(head(data));
% Define the target variable name (column name in the data table)
targetVar = 'Estimated_Lifespan_Years';

% Extract features (X) and target variable (Y)
X = data(:, 2:end-1);  % All columns except Customer_ID and the last one as features
Y = data.(targetVar);  % The target variable column

% Combine features and target into one table for fitlm
data_train = [X, table(Y)];
data_train.Properties.VariableNames{end} = targetVar;

% Split the data into training and testing sets (70% train, 30% test)
cv = cvpartition(height(data), 'HoldOut', 0.3);
trainIdx = training(cv);
testIdx = test(cv);

data_train = data_train(trainIdx, :);
data_test = data(testIdx, :);

% Train and evaluate multiple models
models = {'LinearRegression', 'RandomForest', 'SVM'};
results = table('Size', [3, 3], 'VariableTypes', {'string', 'double', 'double'}, ...
                'VariableNames', {'Model', 'MSE', 'R2'});

for i = 1:length(models)
    modelName = models{i};
    
    switch modelName
        case 'LinearRegression'
            % Specify the formula for Linear Regression
            formula = sprintf('%s ~ %s', targetVar, strjoin(X.Properties.VariableNames, ' + '));
            mdl = fitlm(data_train, formula);
            Y_pred = predict(mdl, data_test(:, 2:end-1));
            
        case 'RandomForest'
            % Train a Random Forest model
            numTrees = 100;
            mdl = TreeBagger(numTrees, data_train(:, 1:end-1), data_train.(targetVar), 'Method', 'regression');
            Y_pred = predict(mdl, data_test(:, 2:end-1));
            
        case 'SVM'
            % Train a Support Vector Machine (SVM) model
            mdl = fitrsvm(data_train(:, 1:end-1), data_train.(targetVar), 'KernelFunction', 'rbf');
            Y_pred = predict(mdl, data_test(:, 2:end-1));
            
        otherwise
            disp('Unknown model.');
            continue;
    end
    
    % Calculate performance metrics
    mse = mean((Y_pred - data_test.(targetVar)).^2);
    r2 = 1 - sum((Y_pred - data_test.(targetVar)).^2) / sum((data_test.(targetVar) - mean(data_test.(targetVar))).^2);
    
    % Store results
    results.Model(i) = modelName;
    results.MSE(i) = mse;
    results.R2(i) = r2;
end

% Display the results
disp(results);

% Plotting MSE
figure;
bar(categorical(results.Model), results.MSE);
xlabel('Model');
ylabel('Mean Squared Error (MSE)');
title('Comparison of Model Performance (MSE)');

% Plotting R-Squared
figure;
bar(categorical(results.Model), results.R2);
xlabel('Model');
ylabel('R-Squared (R^2)');
title('Comparison of Model Performance (R^2)');
