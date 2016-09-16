function output = solve_ghkl(data,Y,lambda,loss,varargin)
% tic;
% n = size(Y,1);  % number of data points

% optional parameters
mingap = 1e-2;                  % duality gap
% maxactive = 300;                % maximum number of active variables
display = 0;                    % 1: if display, 0 otherwise
hull =[];                       % for warm restarts
alpha_hull =[];                      % for warm restarts
b =[];							% for warm restarts
eta =[];						% for warm restarts
% max_difference_active = inf;		% max number of active variables needed to prove optimality
gapeta = 1e-3;                  % gap for eta
% conjgrad = 0;

maxtoadd_nece = 500;              % number of variables added after nece
maxtoadd_suff = 2000;              % number of variables added after suff

myEpsSMO = 1e-6;
myTolEta = 1e-10;
myTolTheta = 1e-8;
mySuffTol = 1e-6;

solver = 'smo_rel';
gap = nan;
current_value = nan;
alphaCell = cell(1,1);
etaCell = cell(1,1);
thetaCell = cell(1,1);
zetaCell = cell(1,1);
objectiveValueArray = [];
activeSetIter = 0;

% solver: 'smo_rel'=> smo (REL-GHKL) (default)
% solver: 'smo'=> smo (GHKL) (general kernel)
% solver: 'shogun'=> shogun (GHKL) (general kernel)

% READ OPTIONAL PARAMETERS
args = varargin;
nargs = length(args);
for i=1:2:nargs
    switch args{i},
        case 'display',        display = args{i+1};
        case 'mingap',        mingap = args{i+1};
        case 'hull',        hull = args{i+1};
        case 'alpha',        alpha_hull = args{i+1};
        case 'b',               b = args{i+1};
        case 'eta',             eta = args{i+1};
        case 'rhonorm',        rhonorm = args{i+1};
		case 'conflictMatrix',   	conflictMatrix = args{i+1};
        case 'maxactive',        maxactive = args{i+1};
%         case 'max_difference_active',        max_difference_active = args{i+1};
        case 'maxtoadd_nece',        maxtoadd_nece = args{i+1};
        case 'maxtoadd_suff',        maxtoadd_suff = args{i+1};
        case 'gapeta',        gapeta = args{i+1};
%         case 'conjgrad',        conjgrad = args{i+1};
        case 'solver', solver = args{i+1};
    end
end

% INITIALIZATION OF HULL WHEN NONE IS GIVEN
if isempty(hull),
    switch data.dag_type
        case 'grid + input_space'
            hull = [ones(1,data.p);ones(data.p,data.p)+eye(data.p)];
%     hull = [];
%     for v=0:3
%         P = perms_reps([1 2],[data.p-v,v]);
%         hull = [hull;P];
%     end
%         case 'mkl + input_space'
%             hull = ones(1,data.p);
%         case 'bimkl + input_space'
%             hull = ones(1,data.p);
    end
end

% SOLVE THE REDUCED HKL PROBLEM
% LOAD DATA FOR THE ENTIRE HULL

[data_hull,sources] = get_reduced_graph_weights_sources(data,hull);

if isempty(eta),
    eta_hull = ones(size(hull,1),1);
else
    eta_hull = eta;
end
% % make sure normalization is correct
eta_hull = eta_hull / ( sum( eta_hull ));

% learn parameters
switch loss
    case 'hinge'

        switch data_hull.dag_type
            case { 'grid + input_space', 'mkl + input_space', 'bimkl + input_space'}
                switch solver
                    case 'smo_rel'
                        [alpha_hull,b_hull,eta_hull,omegaSquared,objectiveValue,theta_kernel,zeta_kernel] = ...
                            solve_ghkl_hinge_smo_rel(alpha_hull,eta_hull,data_hull,Y,lambda,rhonorm);
                    case 'smo'
                        [alpha_hull,b_hull,eta_hull,omegaSquared,objectiveValue,theta_kernel,zeta_kernel] = ...
                            solve_ghkl_hinge_smo_general(eta_hull,data_hull,Y,lambda,rhonorm);
                    case 'shogun'
                        [alpha_hull,b_hull,eta_hull,omegaSquared,objectiveValue,theta_kernel,zeta_kernel] = ...
                            solve_ghkl_hinge_shogun(eta_hull,data_hull,Y,lambda,rhonorm);
                end
        end
end
oldObjectiveValue = objectiveValue;
activeSetIter = activeSetIter+1;
objectiveValueArray(activeSetIter) = objectiveValue;
alphaCell{activeSetIter} = alpha_hull;
etaCell{activeSetIter} = eta_hull;
thetaCell{activeSetIter} = theta_kernel;
zetaCell{activeSetIter} = zeta_kernel;

% check duality gap of the reduced problem (should be small!)
current_value = omegaSquared;
active = find(eta_hull - gapeta/  size(data_hull.X,2)>1e-12);

% START NECESSARY PHASE (only if active is full)
% oldhull = [];
if ( length(eta_hull)>= maxactive )% || ( length(active) <= length(eta_hull) - max_difference_active ),
    necessary_phase = -Inf;
else
    if (length(eta_hull) == length(active) )
        necessary_phase = 1;
%         if display
%             fprintf('\nENTERING NECESSARY PHASE\n\n');
%         end
    else
        necessary_phase = 0;
    end
end

% while  necessary_phase > 0
%     
% %     if display
%         % 		fprintf('number of active variables = %d - number of non zero = %d, gap =%f\n',size(data_hull.hull,1), length(active),gap2 );
% %     end
%     
%     if isempty(sources)
%         % everything is selected: exit!
%         necessary_phase = -Inf;
%         break;
%     end
%     
%     
%     % compute necessary conditions with single added variables
%     gapnec = zeros(size(sources,1),1);
%     for i=1:size(sources,1)
%         [Xloc,weightloc] = get_data_reduced(sources(i,:),data);
%         gapnec(i) = norm( Xloc' * alpha_hull ).^2 * lambda/2;
%         gapnec(i) = gapnec(i) / weightloc / weightloc;
%     end
%     
%     
%     if ~isempty(find(gapnec >= current_value*( 1+ mingap), 1 ));
%         % there are some single sources to add!
%         % compute the ones to add
%         [a,b]=sort(-gapnec);
%         sources = sources(b,:);
%         gapnec = gapnec(b,:);
%         
%         addhull_sources = find(gapnec >= current_value*( 1+ mingap) );
%         % do not include all of them
%         addhull_sources = addhull_sources(1:min(maxtoadd_nece,length(addhull_sources)));
%     else
%         necessary_phase = 0;
%         break;
%     end
%     
%     % prepare new hull for next step
%     if isempty(addhull_sources)
%         necessary_phase = 0;
%         break;
%     end
%     [data_hull,sources] = update_reduced_graph_weights_sources(data,data_hull,sources,addhull_sources,conflictMatrix);
%     hull = data_hull.hull;
%     eta_hull = [eta_hull; zeros(length(addhull_sources),1) ];
%     
%     % learn parameters
%     switch loss
%         case 'hinge'
%             
%             switch data_hull.dag_type
%                 case { 'grid + input_space', 'mkl + input_space', 'bimkl + input_space'}
%                     switch solver
%                         case 'smo_rel'
%                             [alpha_hull,b_hull,eta_hull,omegaSquared,objectiveValue,theta_kernel,zeta_kernel] = ...
%                                 solve_ghkl_hinge_smo_rel(eta_hull,data_hull,Y,lambda,rhonorm);
%                         case 'smo'
%                             [alpha_hull,b_hull,eta_hull,omegaSquared,objectiveValue,theta_kernel,zeta_kernel] = ...
%                                 solve_ghkl_hinge_smo_general(eta_hull,data_hull,Y,lambda,rhonorm);
%                         case 'shogun'
%                             [alpha_hull,b_hull,eta_hull,omegaSquared,objectiveValue,theta_kernel,zeta_kernel] = ...
%                                 solve_ghkl_hinge_shogun(eta_hull,data_hull,Y,lambda,rhonorm);
%                     end
%             end
%             
%     end
%     oldObjectiveValue = objectiveValue;
%     activeSetIter = activeSetIter+1;
%     objectiveValueArray(activeSetIter) = objectiveValue;
%     alphaCell{activeSetIter} = alpha_hull;
%     etaCell{activeSetIter} = eta_hull;
%     thetaCell{activeSetIter} = theta_kernel;
%     zetaCell{activeSetIter} = zeta_kernel;
%     % check duality gap of the reduced problem (should be small!)
%     current_value = omegaSquared;
%     
%     if display
%         fprintf('[solve_ghkl] new eta = %f\n',eta_hull(end))
%     end
%     
%     active = find(eta_hull - gapeta/  size(data_hull.X,2)>1e-12);
% 
%     if ( length(eta_hull)>= maxactive ) || ( length(active) < length(eta_hull) ),
%         if ( length(eta_hull)>= maxactive )
%             necessary_phase = -Inf;
%         else
%             necessary_phase = 0;
%         end
%     end
% end


if ~isinf(necessary_phase),
%     if display
        fprintf('\n[solve_ghkl] ENTERING SUFFICIENT PHASE\n\n');
%     end

    % START SUFFICIENT PHASE
%     oldhull = [];
    sufficient_phase = 1;
    while  sufficient_phase > 0

        active = find(eta_hull - gapeta/  size(data_hull.X,2)>1e-12);
        if display
            fprintf('[solve_ghkl] number of active variables = %d - number of non zero = %d\n',size(data_hull.hull,1), length(active) );
        end
        
        if isempty(sources)
            % everything is selected: exit!
            sufficient_phase = Inf;
            break;
        end
        
        % compute sufficient condition
        switch data_hull.dag_type
            case {'mkl + input_space','mkl + kernels','bimkl + input_space','bimkl + kernels'}
                
                % compute necessary conditions  instead
                gapsuff = zeros(size(sources,1),1);
                for i=1:size(sources,1)
                    [Xloc,weightloc] = get_data_reduced(sources(i,:),data);
                    gapsuff(i) = norm( Xloc' * alpha_hull ).^2 * lambda/2;
                    gapsuff(i) = gapsuff(i) / weightloc / weightloc;
                end
                
            case {'grid + input_space', 'grid + kernels'}
                tic
                gapsuff = check_sufficient_gaps_efficient(sources,alpha_hull,data,lambda);%,hull);
                fprintf('[solve_ghkl] Time in check_sufficient_gaps_efficient: ');
                toc
        end
        
        gap = max(gapsuff)-current_value;
        if (rhonorm ~= 2)
            rhobar = rhonorm/(2-rhonorm);
            rhoTheta = theta_kernel.^rhobar;
            active = find(rhoTheta > myTolTheta);
        else
            active = find(eta_hull > myTolTheta);
        end

%         if display
            fprintf('[solve_ghkl] abs suff gap = %e, rel suff gap = %e violating sources = %d, current_value = %e, active set = %d\n', gap, gap/current_value, ...
                length(find(gapsuff >= current_value*( 1+ mingap))), current_value, length(active));
%         end

        if isnan(gap)
            fprintf('GAP is NAN\n');
            keyboard;
        end

        [a,b]=sort(-gapsuff);
        sources = sources(b,:);
        gapsuff = gapsuff(b);
        addhull_sources =  find(gapsuff >=  current_value*( 1+ mingap) );
        addhull_sources = addhull_sources(1:min(maxtoadd_suff,length(addhull_sources)));
        
        % prepare new hull for next step
        if isempty(addhull_sources)
            sufficient_phase = 0;
            fprintf('\n[solve_ghkl] addhull_sources EMPTY\n');
            break;
        end
        tic
        [data_hull,sources] = update_reduced_graph_weights_sources(data,data_hull,sources,addhull_sources,conflictMatrix);
        fprintf('[solve_ghkl] Time in update_reduced_graph_weights_sources: ');
        toc
        
        hull = data_hull.hull;
        eta_hull = [eta_hull; zeros(length(addhull_sources),1) ];
        
        
        
        % learn parameters
        switch loss
            case 'hinge'
                switch data_hull.dag_type
                    case { 'grid + input_space', 'mkl + input_space', 'bimkl + input_space'}
                        switch solver
                            case 'smo_rel'
                                [alpha_hull,b_hull,eta_hull,omegaSquared,objectiveValue,theta_kernel,zeta_kernel] = ...
                                    solve_ghkl_hinge_smo_rel(alpha_hull,eta_hull,data_hull,Y,lambda,rhonorm);
                            case 'smo'
                                [alpha_hull,b_hull,eta_hull,omegaSquared,objectiveValue,theta_kernel,zeta_kernel] = ...
                                    solve_ghkl_hinge_smo_general(eta_hull,data_hull,Y,lambda,rhonorm);
                            case 'shogun'
                                [alpha_hull,b_hull,eta_hull,omegaSquared,objectiveValue,theta_kernel,zeta_kernel] = ...
                                    solve_ghkl_hinge_shogun(eta_hull,data_hull,Y,lambda,rhonorm);
                        end
                end
        end
        activeSetIter = activeSetIter+1;
        objectiveValueArray(activeSetIter) = objectiveValue;
        alphaCell{activeSetIter} = alpha_hull;
        etaCell{activeSetIter} = eta_hull;
        thetaCell{activeSetIter} = theta_kernel;
        zetaCell{activeSetIter} = zeta_kernel;
        % check duality gap of the reduced problem (should be small!)
        current_value = omegaSquared;
        if abs((oldObjectiveValue-objectiveValue)/objectiveValue) < mySuffTol
            fprintf('\nRelative objective diff low - terminating active set method\n');
            sufficient_phase=0;
        end
        oldObjectiveValue = objectiveValue;        
%         active = find(eta_hull - gapeta/  size(data_hull.X,2)>1e-12);
        
        if size(hull,1)>maxactive, sufficient_phase=0;  end
    end
    
end
% timeCalculated = toc;
% keyboard;

%Pratik new - added on Jan 23 - 2010

output.afterSuffObj = objectiveValue;
fprintf('\nCleaning Phase begins\n');
nodesAfterSufficiency = length(zeta_kernel);
if (any(eta_hull <= myTolEta))
    % get sparsity thru eta - set corresponding theta = 0
    active = find(eta_hull > myTolEta);
    % compute hull
    activeHull = indexActiveHull(active,data_hull.affinity);
    hull = hull(activeHull,:);
    eta_hull = eta_hull(activeHull);
    data_hull = get_reduced_graph_weights_sources(data,hull);
    eta_hull = eta_hull/sum(eta_hull);
    % solve small problem and get new theta
    switch loss
        case 'hinge'
            switch data_hull.dag_type
                case { 'grid + input_space', 'mkl + input_space', 'bimkl + input_space'}
                    switch solver
                        case 'smo_rel'
                            [alpha_hull,b_hull,eta_hull,omegaSquared,objectiveValue,theta_kernel,zeta_kernel] = ...
                                solve_ghkl_hinge_smo_rel(alpha_hull,eta_hull,data_hull,Y,lambda,rhonorm,'myEpsSMO',myEpsSMO);
                        case 'smo'
                            [alpha_hull,b_hull,eta_hull,omegaSquared,objectiveValue,theta_kernel,zeta_kernel] = ...
                                solve_ghkl_hinge_smo_general(eta_hull,data_hull,Y,lambda,rhonorm);
                        case 'shogun'
                            [alpha_hull,b_hull,eta_hull,omegaSquared,objectiveValue,theta_kernel,zeta_kernel] = ...
                                solve_ghkl_hinge_shogun(eta_hull,data_hull,Y,lambda,rhonorm);
                    end
            end
    end
    activeSetIter = activeSetIter+1;
    alphaCell{activeSetIter} = alpha_hull;
    etaCell{activeSetIter} = eta_hull;
    thetaCell{activeSetIter} = theta_kernel;
    zetaCell{activeSetIter} = zeta_kernel;
end
output.afterEtaObj = objectiveValue;
% get sparsity thru theta
if (rhonorm ~= 2)
    rhobar = rhonorm/(2-rhonorm);
    active = zeros(length(theta_kernel),1);
    active(theta_kernel.^rhobar > myTolTheta) = 1;
    theta_kernel = theta_kernel.*active;
    theta_kernel = theta_kernel/norm(theta_kernel,rhobar);
    % recompute alpha thru SVM (as theta is given)
    [alpha_hull, b_hull, newObj] = simpleSVM(data_hull,Y,lambda,theta_kernel,zeta_kernel);
    % final sparsity = # of nodes in nodesAfterSufficiency with theta = 0
    nonZeroNodes = sum(theta_kernel > 0);
    output.newObj = newObj;
else
    % final sparsity = # of nodes in nodesAfterSufficiency with eta = 0(rhonorm = 2 case)
    nonZeroNodes = sum(eta_hull > 0);
end
fprintf('[solve_ghkl] Non-zero nodes after cleaning: %d\n\n', nonZeroNodes);
nodeNumbers = [nodesAfterSufficiency, nonZeroNodes]';

output.alpha=alpha_hull;
output.b =b_hull;
output.eta =eta_hull;
output.theta = theta_kernel;
output.zeta = zeta_kernel;
output.hull=hull;
output.weights = data_hull.weights;
output.descendants = data_hull.affinity;
output.nodeNumbers = nodeNumbers;
output.gap = gap;
output.current_value = current_value;
output.activeSetIter = activeSetIter;
output.alphaCell = alphaCell;
output.etaCell = etaCell;
output.thetaCell = thetaCell;
output.zetaCell = zetaCell;
output.objectiveValueArray = objectiveValueArray;
%Pratik new - added on Jan 23 - 2010



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% function active = compute_hull(hull,active);
% 
% 
% % retake the hull, just to be sure we have not removed
% toadd = [];
% for i=1:size(active,1)
%     toadd = [ toadd; ind2subv(hull(active(i),:),1:prod(hull(active(i),:)))];
% end
% toadd
% toadd = unique([toadd; hull(active,:)],'rows');
% % now maps
% [tf,active]= ismember(toadd,hull,'rows');
% active = sort(active);
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function gapsuff = check_sufficient_gaps_efficient(sources,alpha,data,lambda)%,hull)



switch data.dag_type
    
    
    
    
    case { 'grid + input_space', 'grid + kernels'}
%         p = data.p;
%         n = data.n;
        global K_cache;
        global max_cache;
        
%         global Kdir_cache;
%         global hull_cache;
        global Kdir_caches;
        global source_cache;
        global n_cache;
%         flag = 0;
%         keyboard;
        gapsuff = zeros(1,size(sources,1));
        for t=1:size(sources,1)
            
            tt = int32(sources(t,:))';
            
            %                     temp = repmat(tt,1,size(source_cache,2)) - source_cache;
            %                     distance_to_existing = sum(temp~=0,1)';
            
            % distance_to_existing = find_existing_source_transpose(tt,source_cache);
            % [aold,bold] = min(distance_to_existing);
            [a,b] = find_existing_source_transpose_min_pratik(tt,source_cache);
            % [aold,a]
            %if (a>0) & (mod(t,10)==1), fprintf('%d ',length(find(distance_to_existing==a))); end
            
            if a==0
                % already exactly in the cache
                Knew = K_cache(:,b);
            else
                Knew = K_cache(:,b);
                % only changes the ones that need to be changed
%                 if (size(hull,1) > 45)
%                     keyboard;
%                 end
%                 keyboard;
%                 if (flag == 1)
%                     keyboard;
%                 end
                nonzero = find(tt-source_cache(:,b)~=0);
                negg = find(tt-source_cache(:,b)<0, 1);
                if (~isempty(negg))
                    keyboard;
                end
                for jj =1:length(nonzero);
                    i = nonzero(jj);
%                     Knew = Knew .* Kdir_caches{i}{tt(i)};
%                 end
%                 for jj =1:length(nonzero);
%                     i = nonzero(jj);
                    Knew = (Knew.*Kdir_caches{i}{tt(i)})./Kdir_caches{i}{source_cache(i,b)};
                end                
                n_cache = n_cache + 1;
                storeLoc = mod(n_cache,max_cache)+1;
                if (n_cache > max_cache)
                    fprintf('CACHE EXCEEDED\n');
                end
                if (storeLoc == 1)
                    storeLoc = 2;
                end
                K_cache(:,storeLoc) =  Knew;
                source_cache(:,storeLoc) =  tt;
            end
            gloc =  vectorize_quad_single(Knew,alpha);
            
            gapsuff(t) = gloc * lambda / 2;
            if isnan(gapsuff(t))
                fprintf('GAPSUFF is NAN\n');
                keyboard;
            end
        end
        fprintf('\n[check_sufficient_gaps_efficient] Size, source:%d, source_cache:%d\n', size(sources,1),n_cache);
end

