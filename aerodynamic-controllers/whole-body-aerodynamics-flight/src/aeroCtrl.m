classdef aeroCtrl < matlab.System & matlab.system.mixin.Propagates
    % step_block This block takes as input the joint torques and the
    % applied external forces and evolves the state of the robot
    
    % Public, tunable properties
    properties (Nontunable)
        aero_config;
    end
    
    properties (DiscreteState)
        
    end
    
    properties (Access = private)
        models;
        conditions;
        kinematics;
    end
    
    methods (Access = protected)
        
        function setupImpl(obj)
            obj.models = aeroModel(obj.aero_config);
        end
        
        function aerodynamic_forces = stepImpl(obj, aerodynamicForceAreas, base_velocity, joints_velocity, wind_speed, J_aeroForce, matrixOfAeroTransform)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            obj.set_global_aerodynamic_conditions(wind_speed, base_velocity);
            obj.set_kinematics(J_aeroForce, matrixOfAeroTransform);
            aerodynamic_forces = obj.compute_aerodynamic_forces(base_velocity, joints_velocity, aerodynamicForceAreas);
            % Set to zero aerodynamic effects if not enabled
            if ~obj.models.enable_aero_control
                aerodynamic_forces = 0 * aerodynamic_forces;
            end
        end
        
        function aerodynamic_forces = compute_aerodynamic_forces(obj, base_velocity, joints_velocity, aerodynamicForceAreas)
            % Transform from aerodynamic forces into aerodynamic wrenches
            aerodynamic_forces = obj.compute_aerodynamic_forces_in_wrld_frame(base_velocity, joints_velocity, aerodynamicForceAreas);      
        end
        
        function aerodynamic_forces = compute_aerodynamic_forces_in_wrld_frame(obj, base_velocity, joints_velocity, aerodynamicForceAreas)
            
            if obj.models.use_ctrl_aeroNet
                aerodynamic_forces = 0.5 * obj.conditions.airDensity * norm(obj.conditions.relativeWindVelocity)^2 * aerodynamicForceAreas;
            else
                aerodynamic_forces = nan(3,obj.models.nAeroLinks);
                for i = 1 : obj.models.nAeroLinks
                    linkCoMRelativeWindVelocity  = obj.compute_link_CoM_relative_wind_velocity(obj.models.frameNames{i}, obj.models.linkFrame_T_linkCoM(:,:,i), base_velocity, joints_velocity, obj.conditions.windSpeed);
                    aerodynamic_forces(1:3,i) = obj.compute_link_aerodynamic_force_in_world_frame(obj.models.frameNames{i}, obj.models.frameAxis(:,i), obj.models.linkDiameters(i), ...
                                                                                                  obj.models.linkLengths(i), obj.models.linkReferenceAreas(i), linkCoMRelativeWindVelocity);
                end
            end

        end

        function  link_aerodynamic_force = compute_link_aerodynamic_force_in_world_frame(obj, frameName, frameAxis, linkDiameter, linkLength, linkReferenceArea, linkRelativeWindVelocity)          
            linkAspectRatio    = linkLength/linkDiameter;
            linkAxisVersor     = obj.get_link_aerodynamic_axis_in_world_frame(frameName, frameAxis);
            linkAngleOfAttack  = acosd((transpose(linkAxisVersor) * -linkRelativeWindVelocity) / (norm(linkRelativeWindVelocity) + 1e-9)); % [deg]
            linkReynoldsNumber = (obj.conditions.airDensity * norm(linkRelativeWindVelocity) * linkDiameter) / obj.conditions.airDynamicViscosity;
            
            if obj.models.use_ctrl_zero_order_model
                % Use cfd linear regression model
                [CdA, ~, CnA_bar] = obj.models.get_cfd_model_force_coefficients(frameName, linkAngleOfAttack);
                linkNormalForce = 0.5 * obj.conditions.airDensity * CnA_bar * cross(cross(linkRelativeWindVelocity,linkAxisVersor),linkRelativeWindVelocity);
                linkDragForce   = 0.5 * obj.conditions.airDensity * norm(linkRelativeWindVelocity) * CdA * linkRelativeWindVelocity;
            else
                % Use sphere and cylinder models
                if matches(frameName,'head')
                    [Cd, ~, Cn_sin] = obj.models.get_sphere_force_coefficients(linkReynoldsNumber);
                else
                    [Cd, ~, Cn_sin] = obj.models.get_cylinder_force_coefficients(linkReynoldsNumber, linkAspectRatio, linkAngleOfAttack);
                end
                linkNormalForce = 0.5 * obj.conditions.airDensity * linkReferenceArea * Cn_sin * cross(cross(linkRelativeWindVelocity,linkAxisVersor),linkRelativeWindVelocity);
                linkDragForce   = 0.5 * obj.conditions.airDensity * linkReferenceArea * Cd * norm(linkRelativeWindVelocity) * linkRelativeWindVelocity;
            end
            
            link_aerodynamic_force = linkNormalForce + linkDragForce;
        end
        
        function linkRelativeWindVelocity  = compute_link_CoM_relative_wind_velocity(obj, frameName, linkFrame_T_linkCoM, base_velocity, joints_velocity, windSpeed)
            J_link_frame   = obj.get_frame_jacobian(frameName);

            % computing the jacobian relative to the link CoM from the link
            % frame one
            Link_CoM_pos   = linkFrame_T_linkCoM(1:3,4);
            J_link_CoM_lin = J_link_frame(1:3,:) - wbc.skew(Link_CoM_pos)*J_link_frame(4:6,:);
            J_link_CoM_ang = J_link_frame(4:6,:);
            J_link_CoM     = [J_link_CoM_lin; J_link_CoM_ang];
            
            % computng the link CoM velocity and relative wind velocity
            linkVelocity   = J_link_CoM * [base_velocity; joints_velocity];
            linkRelativeWindVelocity = windSpeed - linkVelocity(1:3);
        end

        function link_axis_versor = get_link_aerodynamic_axis_in_world_frame(obj, frameName, frameAxis)
            w_H_l            = obj.get_frame_transform(frameName);
            link_axis_versor = w_H_l(1:3,1:3) * frameAxis;
        end

        function set_global_aerodynamic_conditions(obj, windSpeed, base_velocity)
            obj.conditions.windSpeed            = windSpeed;
            obj.conditions.relativeWindVelocity = windSpeed - base_velocity(1:3);
            obj.conditions.airDensity           = obj.models.airDensity;
            obj.conditions.airDynamicViscosity  = 1.8e-5;
        end

        function set_kinematics(obj, J_aeroForce, matrixOfAeroTransform)
            obj.kinematics.J_aeroForce            = J_aeroForce;
            obj.kinematics.matrixOfAeroTransform  = matrixOfAeroTransform;
        end

        function J_link_frame = get_frame_jacobian(obj, frameName)
            frameIndex   = obj.get_frame_index(frameName);
            J_link_frame = obj.kinematics.J_aeroForce(6*frameIndex-5:6*frameIndex,:);
        end

        function J_link_frame = get_frame_transform(obj, frameName)
            frameIndex   = obj.get_frame_index(frameName);
            J_link_frame = obj.kinematics.matrixOfAeroTransform(4*frameIndex-3:4*frameIndex,:);
        end

        function frameIndex = get_frame_index(obj, frameName)
            frameIndex = 1;
            while ~matches(obj.models.frameNames{frameIndex},frameName)
                frameIndex = frameIndex + 1;
            end
        end
        
        function [out, out2] = getOutputSizeImpl(~)
            % Return size for each output port
            out = [3 13]; % aerodynamic forces
        end
        
        function [out, out2] = getOutputDataTypeImpl(~)
            % Return data type for each output port
            out = "double";
            out2 = "double";
        end
        
        function [out, out2] = isOutputComplexImpl(~)
            % Return true for each output port with complex data
            out = false;
            out2 = false;
        end
        
        function [out, out2] = isOutputFixedSizeImpl(~)
            % Return true for each output port with fixed size
            out = true;
            out2 = true;
        end
        
        function resetImpl(obj)
            
        end

    end
    
end
