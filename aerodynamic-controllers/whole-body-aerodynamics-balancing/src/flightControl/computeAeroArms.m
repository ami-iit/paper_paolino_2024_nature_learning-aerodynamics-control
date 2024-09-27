function matrixOfAeroArms = computeAeroArms(w_H_LinkFrames, posCoM, aero_config, Config)

    % Distances between the aerodynamic force applied in the link CoM locations 
    % and the robot global CoM position
    n_link_aero      = aero_config.nAeroLinks; % number of aerodynamic links
    matrixOfAeroArms = zeros(3,length(aero_config.linkDiameters));

    for i = 1 : n_link_aero
    r_LinkCoM   = w_H_LinkFrames(1:3,4*i) + aero_config.linkFrame_T_linkCoM(1:3,4,i) - posCoM;
    
    matrixOfAeroArms(:,i) = r_LinkCoM;
    end
end