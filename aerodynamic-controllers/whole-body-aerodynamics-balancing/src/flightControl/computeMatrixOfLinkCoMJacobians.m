function J_aeroForces = computeMatrixOfLinkCoMJacobians(w_H_b, linkFrameJacobians, aero_config)
    
    n_link_aero  = aero_config.nAeroLinks;          % number of aerodynamic links
    J_aeroForces = linkFrameJacobians;              % initialize output

    for i = 1 : n_link_aero
        
        % compute link rotation matrix and extract link CoM pos. in link
        % frame
        w_R_b        = w_H_b(1:3,1:3);
        b_R_Link     = aero_config.linkFrame_T_linkCoM(1:3,1:3,i);
        w_R_Link     = w_R_b*b_R_Link;
        Link_CoM_pos = aero_config.linkFrame_T_linkCoM(1:3,4,i);
    
        % build skew matrix
        S = wbc.skew(w_R_Link*Link_CoM_pos);
        
        % assign linear and angular jacobians for the selected link 
        J_linkFrame_linear  = linkFrameJacobians(6*i-5:6*i-3,:);
        J_linkFrame_angular = linkFrameJacobians(6*i-2:6*i,:);
        
        % linear modification to compute link CoM Jacobian matrix 
        J_aeroForces(6*i-5:6*i-3,:) = J_linkFrame_linear - S * J_linkFrame_angular;

    end
        
end