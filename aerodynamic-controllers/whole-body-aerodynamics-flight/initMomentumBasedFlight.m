%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% /**
%  * Copyright (C) 2018 
%  * @author: Daniele Pucci & Gabriele Nava
%  * Permission is granted to copy, distribute, and/or modify this program
%  * under the terms of the GNU General Public License, version 2 or any
%  * later version published by the Free Software Foundation.
%  *
%  * This program is distributed in the hope that it will be useful, but
%  * WITHOUT ANY WARRANTY; without even the implied warranty of
%  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
%  * Public License for more details
%  */
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear variables
close all
clc

%% GENERAL SIMULATION INFO

robotName = 'iRonCub-Mk1';
setenv('YARP_ROBOT_NAME', robotName)

% Set path to the utility functions and to WBC library
import wbc.*
addpath(genpath('./src/'));
addpath(genpath('./app/'));
addpath('../controlAndDataGui/');
addpath(genpath('../matlab-functions-lib/'));

% Simulation time and delta_t [s]
Config.simulationTime           = 50;
Config.tStep                    = 0.01;

% jets parameters
jets_config.use_jet_dyn         = false;

% aerodynamic simulator parameters and models
aero_config.enable_aero_sim          = true;    % enable simulator aerodynamics
aero_config.use_wind_velocity        = true;    % activate external wind gust
aero_config.use_sim_zero_order_model = false;    % use CFD-based linear regression model
aero_config.use_sim_aeroNet          = true;   % use NN for aerodynamics estimation

% aerodynamic control models
aero_config.enable_aero_control       = true;   % use aero-aware controller
aero_config.use_ctrl_zero_order_model = false;  % use CFD-based linear regression model
aero_config.use_ctrl_aeroNet          = false;  % [WARNING: NOT IMPLEMENTED] use NN for aerodynamics estimation

%% SIMULATION SETTINGS

% Controller type: native GUI, joystick or autopilot
Config.USE_NATIVE_GUI       = false;
Config.USE_FLIGHT_DATA_GUI  = false;
Config.AUTOPILOT_ON         = true;

% Autopilot trajectory (if AUTOPILOT_ON = true)
Config.IN_LINE_TRAJECTORY               = true;
Config.CLOSED_LOOP_TRAJECTORY           = false;
Config.CLOSED_LOOP_TRAJECTORY_NO_TURBO  = false;

% Wind gust profile
Config.CONSTANT_WIND_GUST  = true;
Config.VARIABLE_WIND_GUST  = false;

% Visualizer
confVisualizer.visualizeRobot = true;
confVisualizer.visualizeJets  = false;

% Control type:
%
% Default controller => MOMENTUM BASED CONTROL WITH LYAPUNOV STABILITY (IEEE-RAL)
% If USE_ATTITUDE_CONTROL = true => LINEAR MOMENTUM AND ATTITUDE CONTROL (IEEE-HUMANOIDS)
%
Config.USE_ATTITUDE_CONTROL             = true;

% If Config.INCLUDE_THRUST_LIMITS and/or Config.INCLUDE_JOINTS_LIMITS are
% set to true, the thrusts limits and/or the joints limits are included in
% the control algorithm (as QP constraints)
Config.INCLUDE_THRUST_LIMITS            = true;
Config.INCLUDE_JOINTS_LIMITS            = true;

% Activate visualization and data collection
Config.SCOPE_JOINTS                     = true;
Config.SCOPE_QP                         = true;
Config.SCOPE_COM                        = true;
Config.SCOPE_BASE                       = true;
Config.SCOPE_MOMENTUM                   = true;
Config.SCOPE_JETS                       = true;
Config.SCOPES_WRENCHES                  = true;
Config.SCOPE_GAINS_AND_STATE_MACHINE    = true;
Config.SCOPES_AERODYNAMICS              = true;

% Save data on the workspace after the simulation
Config.SAVE_WORKSPACE                   = true;

%% ADD CONFIGURATION FILES

% Run robot-specific and controller-specific configuration parameters
run(strcat('app/robots/',robotName,'/configRobot.m'));
run(strcat('app/robots/',robotName,'/configAero.m'));
run(strcat('app/robots/',robotName,'/gainsAndParameters.m'));
configJetControlParams;
run(strcat('app/robots/',robotName,'/configJets.m'));

% open the native GUI for control (if no joystick is present)
if Config.USE_NATIVE_GUI
    ironcubControlGui;
end
% Open flight data GUI
if Config.USE_FLIGHT_DATA_GUI
    flightGui = flightDataGui;
end

%% Init simulator core physics paramaters
physics_config.GRAVITY_ACC = [0;0;9.81];
physics_config.TIME_STEP = Config.tStep;
