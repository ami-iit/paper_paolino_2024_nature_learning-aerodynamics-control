classdef aeroDyn < matlab.System & matlab.system.mixin.Propagates
    % step_block This block takes as input the joint torques and the
    % applied external forces and evolves the state of the robot
    
    % Public, tunable properties
    properties (Nontunable)
        aero_config;
        robot_config;
    end
    
    properties (DiscreteState)
        
    end
    
    properties (Access = private)
        models;
        robot;
        conditions;
    end
    
    methods (Access = protected)
        
        function setupImpl(obj)
            obj.models = aeroModel(obj.aero_config);
            obj.robot  = Robot(obj.robot_config);
        end
        
        function [aerodynamic_forces, generalized_aerodynamic_wrench] = stepImpl(obj, aerodynamicForceAreas, base_velocity, joints_velocity, wind_speed)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            obj.set_global_aerodynamic_conditions(wind_speed, base_velocity);
            [aerodynamic_forces, generalized_aerodynamic_wrench] = obj.compute_aerodynamic_forces_and_generalized_aerodynamic_wrench(base_velocity, joints_velocity, aerodynamicForceAreas);
            % Set to zero aerodynamic effects if not enabled
            if ~obj.models.enable_aero_sim
                aerodynamic_forces = 0 * aerodynamic_forces;
                generalized_aerodynamic_wrench = 0 * generalized_aerodynamic_wrench;
            end
        end
        
        function [aerodynamic_forces, generalized_aerodynamic_wrench] = compute_aerodynamic_forces_and_generalized_aerodynamic_wrench(obj, base_velocity, joints_velocity, aerodynamicForceAreas)
            % Transform from aerodynamic forces into aerodynamic wrenches
            aerodynamic_forces = obj.compute_aerodynamic_forces_in_wrld_frame(base_velocity, joints_velocity, aerodynamicForceAreas);
            aerodynamic_wrenches = [aerodynamic_forces; zeros(3, obj.models.nAeroLinks)];
            generalized_aerodynamic_wrench = obj.compute_generalized_aerodynamic_wrench(aerodynamic_wrenches);        
        end
        
        function aerodynamic_forces = compute_aerodynamic_forces_in_wrld_frame(obj, base_velocity, joints_velocity, aerodynamicForceAreas)
            
            if obj.models.use_sim_aeroNet
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
            
            if obj.models.use_sim_zero_order_model
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
            J_link_frame   = obj.robot.get_frame_jacobian(frameName);

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
            w_H_l            = obj.robot.get_frame_H(frameName);
            link_axis_versor = w_H_l(1:3,1:3) * frameAxis;
        end

        function set_global_aerodynamic_conditions(obj, windSpeed, base_velocity)
            obj.conditions.windSpeed            = windSpeed;
            obj.conditions.relativeWindVelocity = windSpeed - base_velocity(1:3);
            obj.conditions.airDensity           = obj.models.airDensity;
            obj.conditions.airDynamicViscosity  = 1.8e-5;
        end
        
        function generalized_aerodynamic_wrench = compute_generalized_aerodynamic_wrench(obj, aerodynamicWrenches)
            generalized_aerodynamic_wrench = zeros(29,1);
            for i = 1 : obj.models.nAeroLinks
                link_gen_aero_wrench = obj.compute_link_generalized_aerodynamic_wrench(obj.models.frameNames{i}, obj.models.linkFrame_X_linkCoM(:,:,i), aerodynamicWrenches(:, i));
                generalized_aerodynamic_wrench = generalized_aerodynamic_wrench + link_gen_aero_wrench;
            end
        end

        function link_gen_aero_wrench = compute_link_generalized_aerodynamic_wrench(obj, frameName, linkFrame_X_linkCoM, aerodynamic_wrench)
            J_link = obj.robot.get_frame_jacobian(frameName);
            linkFrame_aerodynamic_wrench = linkFrame_X_linkCoM * aerodynamic_wrench;
            link_gen_aero_wrench = J_link' * linkFrame_aerodynamic_wrench;
        end
        
        function [out, out2] = getOutputSizeImpl(~)
            % Return size for each output port
            out = [3 13]; % aerodynamic forces
            out2 = [29 1]; % generalized aerodynamic wrench
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