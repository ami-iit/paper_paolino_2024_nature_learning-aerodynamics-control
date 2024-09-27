%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUN THIS SCRIPT TO REMOVE LOCAL PATHS ADDED WHEN RUNNING THE CONTROLLER.
%
% In the Simulink model, this script is run every time the user presses
% the terminate button.

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% close the native control GUI (if present) and reset the GUI input to zero
close all

try
    close(ironcubControlGui);
    set_param('momentumBasedFlight/FLIGHT CONTROLLER (CORE) V3.0/Input_GUI','Value','0') 
catch ME    
    warning(ME.message)
end
    
% remove local paths
rmpath(genpath('./src/'))
rmpath(genpath('./app/'))
rmpath(genpath('../matlab-functions-lib/'));

% Try to remove chache files and folders
try
    rmdir('./slprj','s')  
    delete('momentumBasedFlight.slxc')
catch ME    
    warning(ME.message)
end

% Collect data in a dedicated folder
if Config.SAVE_WORKSPACE

    % Evaluate current date and time to name the folder and experiments
    current_date = char(datetime('now','Format','yyyy-MM-dd'));
    current_time = char(datetime('now','Format','H-mm'));

    % Create the folder (if not existing yet)
    if (~exist(['experiments/',current_date],'dir'))
        mkdir(['experiments/',current_date]);
    end
    
    matFileList = dir(['./experiments/',current_date,'/*.mat']);
   
    save(['./experiments/',current_date,'/exp_',current_time,'.mat'])
end
