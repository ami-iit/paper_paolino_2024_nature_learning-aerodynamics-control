function contactForces_0 = selectInitContactForces_noFT(feetContactIsActive, contactForces, m, Config)

    persistent initContactForces previous_feetInContact
    
    if isempty(initContactForces)
        
        % assumption: initial conditions are ~ 0 for all forces and
        % moments, but the feet vertical forces. This resulted to be a bit
        % more robust than taking the full measured wrench
        initContactForces    = zeros(12,1);
        initContactForces(3)  = contactForces(3);%         (contactForces(3)+contactForces(9))/2;
        initContactForces(9)  = contactForces(9);%         (contactForces(3)+contactForces(9))/2;
        initContactForces(5)  = contactForces(5);%         (contactForces(5)+contactForces(11))/2;
        initContactForces(11) = contactForces(11);%         (contactForces(5)+contactForces(11))/2;

%         (contactForces(3)+contactForces(9))/2;
%         (contactForces(3)+contactForces(9))/2;
%         (contactForces(5)+contactForces(11))/2;
%         (contactForces(5)+contactForces(11))/2;
    end
    if isempty(previous_feetInContact)
        
        previous_feetInContact = 1 - feetContactIsActive;
    end
     
    % in this step, robot moved from flying to balancing (integrator has to
    % be updated with new initial conditions)
    tol = 0.5;
        
    if previous_feetInContact < tol && feetContactIsActive > tol 
        
        % update the contact contact forces initial conditions to the
        % current vertical forces values
        initContactForces(3)  = contactForces(3);
        initContactForces(9)  = contactForces(9);
        initContactForces(5)  = contactForces(5);
        initContactForces(11) = contactForces(11);
    end
        
    % update the previous status of feet in contact
    previous_feetInContact = feetContactIsActive;
    contactForces_0        = initContactForces;
end