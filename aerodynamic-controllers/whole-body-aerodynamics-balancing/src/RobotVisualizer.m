classdef RobotVisualizer < matlab.System & matlab.system.mixin.CustomIcon
    % matlab.System handling the robot visualization
    % go in app/robots/iRonCub*/initVisualizer.m to change the setup config
    
    %@author: Giuseppe L'Erario
    
    properties (Nontunable)
        confVisualizer
        config
        aero_config
    end
    
    properties (DiscreteState)
    end
    
    % Pre-computed constants
    properties (Access = private)
        KinDynModel, viz;
        g   = [0, 0, -9.81]; % gravity vector
        jetFrameList = {'chest_l_jet_turbine', 'chest_r_jet_turbine', 'l_arm_jet_turbine', 'r_arm_jet_turbine'};
        scaling_factor = 0.5;
        linkFrame= {'l_sole','r_sole'};
        aeroFrameNames;
        jointOrder;
        linkFrame_T_linkCoM;
        T_CoM = iDynTree.Transform.Identity(); % CoM frame
        force = iDynTree.Direction();
        jet_int_iDyn = iDynTree.VectorDynSize(4);   
        max_jets_int = 220;
        CoP = iDynTree.Position();
        min_time_viz = 1/20;
    end
    
    methods (Access = protected)
        
        function setupImpl(obj)
            if obj.confVisualizer.visualizeRobot
                % Perform one-time calculations, such as computing constants
                obj.aeroFrameNames = obj.aero_config.frameNames;
                obj.jointOrder = obj.config.jointOrder;
                obj.linkFrame_T_linkCoM = obj.aero_config.linkFrame_T_linkCoM;
                obj.prepareRobot()  % get obj.visualizer
                obj.prepareJets();
                %prepare the initial position of aerodynamics force vector
%                 obj.prepareContactWrench();
                obj.prepareAerodynamicWrench();
                
                tic;
            end
        end
        
        function icon = getIconImpl(~)
            % Define icon for System block
            icon = ["Robot", "Visualizer"];
        end
        
        function stepImpl(obj, world_H_base, joints_positions, jetIntensities, aerodynamic_wrench)
            
            if obj.confVisualizer.visualizeRobot
                iDynTreeWrappers.setRobotState(obj.KinDynModel, world_H_base, joints_positions, zeros(6,1), zeros(23,1), obj.g);
                if obj.viz.run()
                    time_interval = toc;
                    if time_interval > obj.min_time_viz 
                        obj.updateVisualization(world_H_base, joints_positions);
                        obj.updateJets(jetIntensities);
%                         obj.updateContactWrench(contact_wrench_left_right);
                        obj.updateAerodynamicWrench(aerodynamic_wrench);
                        obj.viz.draw();
                        tic
                    end
                else
                    error('Closing visualizer.')
                end
            end
        end
        
        
        function updateVisualization(obj, world_H_base, joints_positions)
                s = iDynTree.VectorDynSize(obj.KinDynModel.NDOF);     
                for k = 0:length(joints_positions)-1
                    s.setVal(k,joints_positions(k+1));
                end
                baseRotation_iDyntree = iDynTree.Rotation();
                baseOrigin_iDyntree   = iDynTree.Position();
                T = iDynTree.Transform();
                for k = 0:2
                    baseOrigin_iDyntree.setVal(k,world_H_base(k+1,4));
                    for j = 0:2
                        baseRotation_iDyntree.setVal(k,j,world_H_base(k+1,j+1));                   
                    end
                end      
                T.setRotation(baseRotation_iDyntree);
                T.setPosition(baseOrigin_iDyntree);
                obj.viz.modelViz('iRonCub').setPositions(T, s);
                % CoM frame update
                obj.T_CoM.setPosition(obj.KinDynModel.kinDynComp.getCenterOfMassPosition());
                obj.viz.frames().updateFrame(0, obj.T_CoM);
        end
        
        
        function prepareRobot(obj)
            % Main variable of iDyntreeWrappers used for many things including updating
            % robot position and getting world to frame transforms
            % create KinDyn model
            obj.KinDynModel = iDynTreeWrappers.loadReducedModel(obj.jointOrder, 'root_link', ...
                obj.config.modelPath, obj.config.fileName, false);
            % instantiate iDynTree visualizer
            obj.viz = iDynTree.Visualizer();
            obj.viz.init();
            % if you're courious compile idyntree in devel and uncommend the line below
            % obj.viz.setColorPalette('meshcat');
            % add 'iRonCub' robot the the visualizer
            obj.viz.addModel(obj.KinDynModel.kinDynComp.model(), 'iRonCub');
            env = obj.viz.enviroment();
            env.setElementVisibility('floor_grid', true);
            env.setElementVisibility('world_frame', true);
            obj.viz.camera().animator().enableMouseControl(true);
            % adding lights
            obj.viz.enviroment().addLight('sun1');
            obj.viz.enviroment().lightViz('sun1').setType(iDynTree.DIRECTIONAL_LIGHT);
            obj.viz.enviroment().lightViz('sun1').setDirection(iDynTree.Direction(-1, 0, 0));
            obj.viz.enviroment().addLight('sun2');
            obj.viz.enviroment().lightViz('sun2').setType(iDynTree.DIRECTIONAL_LIGHT);
            obj.viz.enviroment().lightViz('sun2').setDirection(iDynTree.Direction(1, 0, 0));
%             obj.viz.modelViz('iRonCub').setFeatureVisibility('wireframe', true);
            % add CoM frame
            obj.viz.frames().addFrame(iDynTree.Transform.Identity(), 0.2);
        end
        
        function prepareJets(obj)
            % setting jets
            orange = iDynTree.ColorViz(1.0, 0.6, 0.1, 0.0);
            obj.viz.modelViz('iRonCub').jets().setJetColor(0, orange);
            obj.viz.modelViz('iRonCub').jets().setJetColor(1, orange);
            obj.viz.modelViz('iRonCub').jets().setJetColor(2, orange);
            obj.viz.modelViz('iRonCub').jets().setJetColor(3, orange);
            obj.viz.modelViz('iRonCub').jets().setJetsFrames(obj.jetFrameList);
            obj.viz.modelViz('iRonCub').jets().setJetsDimensions(0.02, 0.1, 0.3);
            obj.viz.modelViz('iRonCub').jets().setJetDirection(0, iDynTree.Direction(0, 0, 1.0));
            obj.viz.modelViz('iRonCub').jets().setJetDirection(1, iDynTree.Direction(0, 0, 1.0));
            obj.viz.modelViz('iRonCub').jets().setJetDirection(2, iDynTree.Direction(0, 0, 1.0));
            obj.viz.modelViz('iRonCub').jets().setJetDirection(3, iDynTree.Direction(0, 0, 1.0));
        end
        
        function prepareContactWrench(obj) 
            % prepare forces plotting            
            for i=1:length(obj.linkFrame)
                linkTransform = obj.KinDynModel.kinDynComp.getWorldTransform(obj.linkFrame{i});
                for j=1:3
                    % note that the indexing starts from 0 (not from 1) as
                    % in C++
                    obj.force.setVal(j-1, 0);
                end
                obj.viz.vectors().addVector(linkTransform.getPosition(), obj.force);
            end
            % prepare also a flat arrow for the CoP (the last one)
            obj.viz.vectors().addVector(linkTransform.getPosition(), obj.force);
        end 

        function prepareAerodynamicWrench(obj) 
            % prepare forces plotting            
            for i=1:length(obj.aeroFrameNames)
                linkTransform = obj.KinDynModel.kinDynComp.getWorldTransform(obj.aeroFrameNames{i});
                CoMTransform  = linkTransform.asHomogeneousTransform().toMatlab() * obj.aero_config.linkFrame_T_linkCoM(:,:,i);
                vectorPos     = iDynTree.Position(CoMTransform(1,4),CoMTransform(2,4),CoMTransform(3,4));
%                 vectorPos     = linkTransform.getPosition();
                for j=1:3
                    % note that the indexing starts from 0 (not from 1) as
                    % in C++
                    obj.force.setVal(j-1, 0);
                end
                obj.viz.vectors().addVector(vectorPos, obj.force);
            end
            % prepare also a flat arrow for the CoP (the last one)
%             obj.viz.vectors().addVector(linkTransform.getPosition(), obj.force);
        end 
        
        function updateJets(obj, jetIntensities)
            for i=1:4
                obj.jet_int_iDyn.setVal(i-1, jetIntensities(i)/obj.max_jets_int);
            end
            obj.viz.modelViz('iRonCub').jets().setJetsIntensity(obj.jet_int_iDyn);
        end
        
        
        function updateContactWrench(obj, contactWrench)
            CoPs = zeros(3,2);
            %update the contact forces 
            for i=1:length(obj.linkFrame)
                linkTransform = obj.KinDynModel.kinDynComp.getWorldTransform(obj.linkFrame{i});
                for j=1:3
                    % the single aerodynamics force is scaled by a constant factor
                    obj.force.setVal(j-1, contactWrench(j, i) * obj.scaling_factor);
                end
                obj.viz.vectors().updateVector(i-1, linkTransform.getPosition(), obj.force);
                % cop computation
                p = linkTransform.getPosition().toMatlab();
                % force-torque balance 
                CoPs(:,i) = [[contactWrench(5,i); -contactWrench(4,i)] / contactWrench(3,i); 0.0] + [p(1:2); 0.0];
            end
            % set contribution of the single CoP to zero if the contact
            % force = 0. Also, if the one force is non null and the other
            % is null do not compute the mean
            CoPs(:,1) = CoPs(:,1) * (contactWrench(3,1) > 0.0) * (1 - 0.5 * (contactWrench(3,2) > 0.0)); 
            CoPs(:,2) = CoPs(:,2) * (contactWrench(3,2) > 0.0) * (1 - 0.5 * (contactWrench(3,1) > 0.0)); 
            cop = sum(CoPs,2);
            for j=1:3
                % the single aerodynamics force is scaled by a constant factor
                obj.CoP.setVal(j-1, cop(j));
                obj.force.setVal(j-1, 0.0);
            end
            obj.viz.vectors().updateVector(i, obj.CoP, obj.force);
        end

        function updateAerodynamicWrench(obj, aerodynamicWrench)
%             CoPs = zeros(3,2);
            %update the contact forces 
            for i=1:length(obj.aeroFrameNames)
                linkTransform = obj.KinDynModel.kinDynComp.getWorldTransform(obj.aeroFrameNames{i});
                CoMTransform  = linkTransform.asHomogeneousTransform().toMatlab() * obj.aero_config.linkFrame_T_linkCoM(:,:,i);
                vectorPos     = iDynTree.Position(CoMTransform(1,4),CoMTransform(2,4),CoMTransform(3,4));
%                 vectorPos     = linkTransform.getPosition();
%                 for j=1:3
%                     % the single aerodynamics force is scaled by a constant factor
%                     obj.force.setVal(j-1, aerodynamicWrench(j, i) * obj.scaling_factor);
%                 end
                obj.force.fromMatlab(aerodynamicWrench(:, i) * obj.scaling_factor);
                obj.viz.vectors().updateVector(i-1, vectorPos, obj.force);
                % cop computation
%                 p = linkTransform.getPosition().toMatlab();
                % force-torque balance 
%                 CoPs(:,i) = [[aerodynamicWrench(5,i); -aerodynamicWrench(4,i)] / aerodynamicWrench(3,i); 0.0] + [p(1:2); 0.0];
            end
            % set contribution of the single CoP to zero if the contact
            % force = 0. Also, if the one force is non null and the other
            % is null do not compute the mean
%             CoPs(:,1) = CoPs(:,1) * (aerodynamicWrench(3,1) > 0.0) * (1 - 0.5 * (aerodynamicWrench(3,2) > 0.0)); 
%             CoPs(:,2) = CoPs(:,2) * (aerodynamicWrench(3,2) > 0.0) * (1 - 0.5 * (aerodynamicWrench(3,1) > 0.0)); 
%             cop = sum(CoPs,2);
%             for j=1:3
%                 % the single aerodynamics force is scaled by a constant factor
%                 obj.CoP.setVal(j-1, cop(j));
%                 obj.force.setVal(j-1, 0.0);
%             end
%             obj.viz.vectors().updateVector(i, obj.CoP, obj.force);
        end
    end
end