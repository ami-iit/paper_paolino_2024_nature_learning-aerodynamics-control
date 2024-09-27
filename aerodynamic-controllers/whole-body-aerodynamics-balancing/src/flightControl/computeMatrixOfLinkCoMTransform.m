function matrixOfLinkCoMTransform = computeMatrixOfLinkCoMTransform(w_H_LinkFrames, posCoM, aero_config, Config)

    % Distances between the aerodynamic force applied in the link CoM locations 
    % and the robot global CoM position
    n_link_aero      = aero_config.nAeroLinks; % number of aerodynamic links
    matrixOfLinkCoMTransform = zeros(4*n_link_aero,4);

    for i = 1 : n_link_aero

        w_H_LinkCoM      = w_H_LinkFrames(:,4*i-3:4*i)*aero_config.linkFrame_T_linkCoM(:,:,i);

        matrixOfLinkCoMTransform(4*i-3:4*i,:) = w_H_LinkCoM;

    end

end