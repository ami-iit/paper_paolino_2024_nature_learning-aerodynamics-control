function w_aerodynamicForceAreas = transformAeroNetOutput(w_H_b, aerodynamicForceAreas)

    % Transform aerodynamic force areas data format
    aerodynamicForceAreas = double(aerodynamicForceAreas);
    aerodynamicForceAreas = reshape(aerodynamicForceAreas,[13,3]);
    aerodynamicForceAreas = transpose(aerodynamicForceAreas);
    
    % Invert ClA and CsA (aeroNet gives {CdA, ClA, CsA})
    aerodynamicDragAreas = aerodynamicForceAreas(1,:);
    aerodynamicLiftAreas = aerodynamicForceAreas(2,:);
    aerodynamicSideAreas = aerodynamicForceAreas(3,:);
    
    aerodynamicForceAreas = [aerodynamicDragAreas; ...
                             aerodynamicSideAreas; ...
                             aerodynamicLiftAreas];

    % Change aerodynamic force areas reference frame from base to world
    w_R_b = w_H_b(1:3,1:3);
    w_aerodynamicForceAreas = w_R_b * aerodynamicForceAreas;

end