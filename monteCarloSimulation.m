function monteCarloPredictions = monteCarloSimulation(randomForestModel, X, numSimulations)
    % Define distributions for each input feature based on mean and std
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

    % Calculate 95% Confidence Interval for Monte Carlo predictions
    CI_MonteCarlo = prctile(monteCarloPredictions, [2.5 97.5]);
    disp(['95% Confidence Interval for Predicted Battery Life (Monte Carlo): ', num2str(CI_MonteCarlo(1)), ' - ', num2str(CI_MonteCarlo(2))]);
        % Analyze the results: plot histogram of the predicted battery life
    figure;
    histogram(monteCarloPredictions, 30);
    hold on;
    xline(CI_MonteCarlo(1), 'r--', 'LineWidth', 2);  % Lower bound of 95% CI for Monte Carlo
    xline(CI_MonteCarlo(2), 'r--', 'LineWidth', 2);  % Upper bound of 95% CI for Monte Carlo
    xlabel('Predicted Battery Life (Years)');
    ylabel('Frequency');
    title('Monte Carlo Simulation of Predicted Battery Life');
end
