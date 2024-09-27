close all;
clear all;
clc;

%% INPUT
% Name of the experiment
experimentName = 'exp_17-27'; % exp_17-32: front wind gust | exp_17-27: lateral wind gust
% Plot file extension
imgFormat = 'svg';

%% Load mat file
experimentFullPath = ['../../experiments19-Oct-2023/',experimentName,'.mat'];
load(experimentFullPath);

%% Create experiments plots directory
% Plots storage directory
experimentPlotDir = ['experimentPlots/',experimentName];

if (~exist(experimentPlotDir,'dir'))
    mkdir(experimentPlotDir);
end

%% Plot data
% Define the variables to be plotted
variableNames = {'jointVelDes_SCOPE', ... x
                 'jointPosMeas_SCOPE', ...x
                 'jointVelMeas_SCOPE', ...x
                 'desWrenchLFoot_SCOPE', ...x
                 'desWrenchRFoot_SCOPE', ...x
                 'LFoot_fDotDes_SCOPE', ...x
                 'RFoot_fDotDes_SCOPE', ...x
                 'linMom_des_SCOPE', ...
                 'LDotEstLinear_SCOPE', ...x
                 'LDotEstAngular_SCOPE', ...x
                 'posCOM_des_SCOPE', ...
                 'posCoM_SCOPE', ...x
                 'velCoM_SCOPE', ...x
                 'basePos_SCOPE', ...x
                 'baseVel_SCOPE', ...x
                 'windSpeed_SCOPE', ...
                 ...'aerodynamic_forces_SCOPE', ...
                 'jointTorqueDes_SCOPE', ...x
                 'jointTorqueMeas_SCOPE', ...x
                 'sDDot_SCOPE', ... torque QP residuals (sDDot_meas-sDDot(u*))
                 'wrench_Lfoot_torque_QP_SCOPE', ...x
                 'wrench_Rfoot_torque_QP_SCOPE', ...x
                 'leftFootExtWrench_SCOPE', ...x
                 'rightFootExtWrench_SCOPE'}; ...x

% Loop for plotting a variable in each figure
for variableIn = 1 : length(variableNames)
    
    % Assign variable name
    variableName = variableNames{variableIn};
    figureName   = variableName(1:end-6);
    
    % Check variable values number of dimensions
    varDimNum = ndims(out.(variableName).signals(1).values);
    if varDimNum > 2
        changeFormat = true;
        disp(['[',figureName,'] resized for plotting'])
    else
        changeFormat = false;
    end
    
    % Initialize figure
    fig = figure('NumberTitle', 'off', 'Name', figureName);
    fig.Visible  = "off";
    fig.Position = [0 0 1920 1440];
    subplotNumber = length(out.(variableName).signals);

    % Loop for plotting the signals in each subplot
    for subplotIn = 1 : subplotNumber

        % Assigning variables
        time      = out.(variableName).time;
        values    = out.(variableName).signals(subplotIn).values;
        plotTitle = out.(variableName).signals(subplotIn).label;
        if (changeFormat) values = squeeze(values); end

        % Creating the sublplot
        subplot(subplotNumber,1,subplotIn)
        plot(time,values);
        title(plotTitle,'Interpreter','none');
        grid on;
        
        % Assigning legend lables
        labelsNum = length(out.(variableName).signals(subplotIn).plotStyle);
        labels = cell(1,labelsNum);
        for legendIn = 1:labelsNum
            labels{legendIn} = ['signal',num2str(legendIn)];
        end
        legend(labels);
        
    end

    saveas(fig,[experimentPlotDir,'/',figureName],imgFormat)
end


%% Link aerodynamic forces 

% Assign variable name
variableName = 'aerodynamic_forces_SCOPE';
figureName   = variableName(1:end-6);
subplotNumbers = [4,4,5];
partIndex = 0; % initialize
for aeroForcePlotIn = 1 : 3

    % Initialize figure 1
    fig = figure('NumberTitle', 'off', 'Name', figureName);
    fig.Visible  = "off";
    fig.Position = [0 0 1920 1440];
    subplotNumber = subplotNumbers(aeroForcePlotIn);

    % Loop for plotting the signals in each subplot
    for subplotIn = 1 : subplotNumber

        partIndex = partIndex + 1;
        % Assigning variables
        time      = out.(variableName).time;
        values    = out.(variableName).signals.values(:,partIndex,:);
        values    = squeeze(values);
        plotTitle = aero_config_with_frames.frameNames{partIndex};

        % Creating the sublplot
        subplot(subplotNumber,1,subplotIn)
        plot(time,values);
        title(plotTitle,'Interpreter','none');
        grid on;

        % Assigning legend lables
        labelsNum = 3;
        labels = cell(1,labelsNum);
        for legendIn = 1:labelsNum
            labels{legendIn} = ['signal',num2str(legendIn)];
        end
        legend(labels);

    end

    saveas(fig,[experimentPlotDir,'/',figureName,num2str(aeroForcePlotIn)],imgFormat)
end


%% Total aerodynamic force

% Assign variable name
variableName = 'aerodynamic_forces_SCOPE';
figureName   = variableName(1:end-6);
partIndex = 0; % initialize

% Initialize figure 1
fig = figure('NumberTitle', 'off', 'Name', figureName);
fig.Visible  = "off";
fig.Position = [0 0 1920 1440];
subplotNumber = 3;
plotTitles = {'Fx', 'Fy', 'Fz'};

% Loop for plotting the signals in each subplot
for subplotIn = 1 : subplotNumber

    % Assigning variables
    time      = out.(variableName).time;
    values    = out.(variableName).signals.values(subplotIn,:,:);
    values    = squeeze(values);
    values    = sum(values);
    plotTitle = plotTitles{subplotIn};

    % Creating the sublplot
    subplot(subplotNumber,1,subplotIn)
    plot(time,values);
    title(plotTitle,'Interpreter','none');
    grid on;

end

saveas(fig,[experimentPlotDir,'/',figureName,'_total'],imgFormat)




