function [randomForestModel, YPred_RF, mseTest_RF] = trainAndTestRandomForest(dataFile)
    % Load the data
    data = readtable(dataFile);

    % Prepare the input features and target variable
    features = data(:, {'Screen_Time_Hours', 'Charging_Cycles_Per_Week', 'Bluetooth_Hours', ...
                        'Cellular_Data_Hours', 'GPS_Hours', 'Ambient_Temperature_Celsius', ...
                        'Battery_Capacity_mAh'});
    target = data.Estimated_Lifespan_Years;

    % Convert the table to arrays for machine learning
    X = table2array(features);
    Y = target;

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
end
