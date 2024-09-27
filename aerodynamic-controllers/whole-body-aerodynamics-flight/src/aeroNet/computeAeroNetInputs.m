function [b_relWindDir, jointPosDeg] = computeAeroNetInputs(w_H_b, base_velocity, joints_position, windSpeed)

    %% Compute relative wind versor in body frame (by def. equal to b_x_A)
    w_R_b = w_H_b(1:3,1:3);
    b_R_w = transpose(w_R_b);
    relativeWindVelocity   = windSpeed - base_velocity(1:3);

    if norm(relativeWindVelocity) ~= 0
        b_relativeWindVelocity = b_R_w * relativeWindVelocity;
        b_relWindDir = b_relativeWindVelocity / norm(b_relativeWindVelocity);
    else
        b_relWindDir = [1;0;0];
    end

    b_relWindDir = transpose(b_relWindDir);

    %% Convert joints position from rad to deg
    jointPosRad = joints_position([1:15 18:end-2]);
    jointPosDeg = transpose(jointPosRad) *180/pi;

end