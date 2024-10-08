function windVelocity = generateWindVelocity(maxWindVel, time, Config)
    
    persistent windVel
    
    windVel = zeros(3,1);
    
    
    % Wind gust time parameters
    t_rampStart = 5;
    t_rampEnd   = 10;
    
    % t_rotStart  = 7;
    % t_rotEnd    = 27;

    
    % Generate constant wind gust profile
    if Config.CONSTANT_WIND_GUST

        if (time >= t_rampStart && time <= t_rampEnd) 
            windVel = maxWindVel * ...
                (time - t_rampStart)/(t_rampEnd - t_rampStart); % ramp up
        end
        if time >= t_rampEnd     
            windVel = maxWindVel;   % max const value reached
        end
    
    % Generate variable wind gust profile
    elseif Config.VARIABLE_WIND_GUST
        windVel = zeros(3,1);
    end
    
    % Assigning windVelocity
    windVelocity = windVel;
    



    % % Wind Velocity rotation on z axis
    % if (time >  t_rotStart && time <= t_rotEnd) 
    %     rotPeriod = t_rotEnd - t_rotStart;
    %     rotOmega  = 360/rotPeriod;
    %     rotAngle  = rotOmega*(time-t_rotStart);
    %     rotMatrix = rotz(rotAngle);
    %     windVelocity = rotMatrix*maxWindVelocity;
    % end

end