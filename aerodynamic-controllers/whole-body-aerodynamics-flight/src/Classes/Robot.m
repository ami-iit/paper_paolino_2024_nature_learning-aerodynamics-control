classdef Robot < handle
    %ROBOT The Robot class exploits the iDynTree wrappers to compute Kinematic and Dynamic quantities.
    % Robot Methods:
    %    set_robot_state - Sets the robot state with kinematic information
    %    get_mass_matrix - Returns the mass matrix
    %    get_bias_forces - Returns the bias force
    %    get_feet_jacobians - Returns the Jacobians of the feet
    %    get_feet_JDot_nu - Returns the Jacobian derivative of the feet multiplied by the configuration velocity
    %    get_feet_H - Returns the Homogenous transform of the feet in the world frame
    %    get_frame_jacobian - Returns the Jacobian of a specified frame
    %    get_frame_JDot_nu - Returns the Jacobian derivative of a specified frame multiplied by the configuration velocity
    %    forward_dynamics - Compute forward dynamics

    properties
        NDOF; % DOF of the robot
        
    end

    properties (Access = private)
        KinDynModel;% kynDyn robot model
        g = [0, 0, -9.81]; % gravity vector
        M_iDyn; % mass matrix iDynTree
        J_LFoot_iDyntree; % Jacobian relative to left foot
        J_RFoot_iDyntree; % Jacobian relative to right foot
        JDot_nu_LFoot_iDyntree; % \dot{J} \nu relative to left foot
        JDot_nu_RFoot_iDyntree; % \dot{J} \nu relative to right foot
        LFoot_frameID; RFoot_frameID; % framesID relative to left and right foot
        h_iDyn; % bias forces iDynTree
        S; % selector matrix
    end

    methods

        function obj = Robot(config)
            % ROBOT Sets up the object. Takes as input a config file
            % loading the model
            obj.KinDynModel = iDynTreeWrappers.loadReducedModel(config.jointOrder, 'root_link', ...
                                                                config.modelPath, config.fileName, false);

            %initialize robot state
            obj.set_robot_state(config.initialConditions.w_H_b, config.initialConditions.s, ...
                                config.initialConditions.base_pose_dot, config.initialConditions.s_dot);

            % initialize general quantites and iDynTree objects
            obj.J_LFoot_iDyntree = iDynTree.MatrixDynSize(6, obj.KinDynModel.NDOF + 6);
            obj.J_RFoot_iDyntree = iDynTree.MatrixDynSize(6, obj.KinDynModel.NDOF + 6);
            obj.LFoot_frameID = obj.KinDynModel.kinDynComp.getFrameIndex('l_sole');
            obj.RFoot_frameID = obj.KinDynModel.kinDynComp.getFrameIndex('r_sole');
            obj.h_iDyn = iDynTree.FreeFloatingGeneralizedTorques(obj.KinDynModel.kinDynComp.model);
            obj.M_iDyn = iDynTree.MatrixDynSize();
            obj.NDOF = obj.KinDynModel.NDOF;
            obj.S = [zeros(6, obj.KinDynModel.NDOF); eye(obj.KinDynModel.NDOF)];
        end

        function set_robot_state(obj, w_H_b, s, base_pose_dot, s_dot)
            % set_robot_state Sets the robot state with kinematic information
            % INPUT: - w_H_b: [4,4] Homogeneous transformation body to world frame
            %        - s = [NDOF, 1] Joints position vector
            %        - base_pose_dot = [6,1] linear and angular velocity of the base
            %        - s_dot = [NDOF, 1] Joints velocity vector
            iDynTreeWrappers.setRobotState(obj.KinDynModel, w_H_b, s, ...
                                           base_pose_dot, s_dot, obj.g);
        end

        function M = get_mass_matrix(obj)
            % get_mass_matrix Returns the mass matrix
            % OUTPUT: - M: mass matrix
            if (~obj.KinDynModel.kinDynComp.getFreeFloatingMassMatrix(obj.M_iDyn))
                error('[Robot: get_mass_matrix] Unable to retrieve the mass matrix');
            end

            M = obj.M_iDyn.toMatlab;
        end

        function h = get_bias_forces(obj)
            % get_bias_forces Returns the bias force
            % OUTPUT: - h: bias force
            if (~obj.KinDynModel.kinDynComp.generalizedBiasForces(obj.h_iDyn))
                error('[Robot: get_bias_forces] Unable to retrieve the bias force')
            end

            h_b = obj.h_iDyn.baseWrench.toMatlab;
            h_s = obj.h_iDyn.jointTorques.toMatlab;
            h = [h_b; h_s];
        end

        function [J_LFoot, J_RFoot] = get_feet_jacobians(obj)
            % get_feet_jacobians Returns the Jacobians of the feet
            % OUTPUT: - J_left_foot: Jacobian of the left foot
            %         - J_right_foot: Jacobian of the right foot
            if (~obj.KinDynModel.kinDynComp.getFrameFreeFloatingJacobian(obj.LFoot_frameID, obj.J_LFoot_iDyntree))
                error('[Robot: get_feet_jacobians] Unable to retrieve the left foot jacobian');
            end

            if (~obj.KinDynModel.kinDynComp.getFrameFreeFloatingJacobian(obj.RFoot_frameID, obj.J_RFoot_iDyntree))
                error('[Robot: get_feet_jacobians] Unable to retrieve the left foot jacobian');
            end

            J_LFoot = obj.J_LFoot_iDyntree.toMatlab;
            J_RFoot = obj.J_RFoot_iDyntree.toMatlab;
        end

        function [JDot_nu_LFOOT, JDot_nu_RFOOT] = get_feet_JDot_nu(obj)
            % get_feet_JDot_nu Returns the Jacobian derivative of the feet multiplied by the configuration velocity
            % OUTPUT: - JDot_nu_LFOOT: \dot{J} nu relative to the left foot
            %         - JDot_nu_RFOOT: \dot{J} nu relative to the right foot
            obj.JDot_nu_LFoot_iDyntree = obj.KinDynModel.kinDynComp.getFrameBiasAcc('l_sole');
            obj.JDot_nu_RFoot_iDyntree = obj.KinDynModel.kinDynComp.getFrameBiasAcc('r_sole');
            JDot_nu_LFOOT = obj.JDot_nu_LFoot_iDyntree.toMatlab;
            JDot_nu_RFOOT = obj.JDot_nu_RFoot_iDyntree.toMatlab;
        end

        function [H_LFOOT, H_RFOOT] = get_feet_H(obj)
            % get_feet_H Returns the Homogenous transform of the feet in the world frame
            % OUTPUT: - H_LFOOT: w_H_b of relative to the left feet
            %         - H_RFOOT: w_H_b of relative to the right feet
            H_LFOOT = iDynTreeWrappers.getWorldTransform(obj.KinDynModel, 'l_sole');
            H_RFOOT = iDynTreeWrappers.getWorldTransform(obj.KinDynModel, 'r_sole');
        end
        
        function [w_H_b] = get_frame_H(obj, frame)
            % get_frame_H Returns the Homogenous transform of the frame in the world frame
            % OUTPUT: - w_H_b of relative to the frame
            w_H_b = iDynTreeWrappers.getWorldTransform(obj.KinDynModel, frame);
        end
        function J_com=get_com_jacobian(obj)
            J_com=iDynTreeWrappers.getCenterOfMassJacobian(obj.KinDynModel);
        end

        function J = get_frame_jacobian(obj, frame)
            % get_frame_jacobian Returns the Jacobian of a specified frame
            % INPUT: - frame: the frame string
            % OUTPUT: j: the Jacobian of the frame
            J_iDyntree = iDynTree.MatrixDynSize(6, obj.KinDynModel.NDOF + 6);
            frame_id = obj.KinDynModel.kinDynComp.getFrameIndex(frame);

            if (~obj.KinDynModel.kinDynComp.getFrameFreeFloatingJacobian(frame_id, J_iDyntree))
                error(['[Robot: get_frame_Jacobian] Unable to retrieve the ', frame, ' jacobian']);
            end

            J = J_iDyntree.toMatlab;
        end

        function JDot_nu = get_frame_JDot_nu(obj, frame)
            % get_frame_JDot_nu Returns the Jacobian derivative of a specified frame multiplied by the configuration velocity
            % INPUT: - frame: the frame string
            % OUTPUT: J: \dot{J} nu of the frame
            JDot_nu_iDyntree = obj.KinDynModel.kinDynComp.getFrameBiasAcc(frame);
            JDot_nu = JDot_nu_iDyntree.toMatlab;
        end

        function [base_pose_ddot, s_ddot] = forward_dynamics(obj, torque, generalized_total_wrench)
            % forward_dynamics Compute forward dynamics
            % \dot{v} = inv{M}(S*tau + generalized_external_forces - h)
            % INPUT: - torque: the joints torque
            %        - generalized_total_wrench: the sum of the external wrenches in the configuration space
            % OUTPUT: - base_pose_ddot: the linear and angular acceleration of the base
            %         - s_ddot: the joints acceleration
            M = obj.get_mass_matrix();
            h = obj.get_bias_forces();
            ddot = M \ (obj.S * torque + generalized_total_wrench - h);
            base_pose_ddot = ddot(1:6);
            s_ddot = ddot(7:end);
        end
    end
end
