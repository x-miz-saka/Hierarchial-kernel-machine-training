function [alphaOpt,bOpt,eta,omegaSquared,finalLpDualObj,theta,zeta] = solve_ghkl_hinge_smo_rel(alpha_smo,eta,data,Y,lambda,rho,varargin)

% solve the milasso for a given regularization parameter
n = size(data.X,1);
p =length(data.affinity);
affs = data.affinity;
weights = data.weights;
fprintf('\n[solve_ghkl_hinge_smo_rel] No. of nodes: %d\n',p);
X = data.X;	% no need to center here for hinge
% if ~all(sum(X))
%     fprintf('[solve_ghkl_hinge_smo_rel] Zero feature occurred\n');
%     keyboard;
% end
w1 = sum(Y == 1);
w2 = sum(Y == -1);
if (w1+w2 ~= n)
    fprintf('[solve_ghkl_hinge_smo_rel] w1+w2~=1\n');
    keyboard;
end
details.balance = 1;
details.wpos = w2/n;
details.wneg = w1/n;
details.numTrain = n;
yKy = 0;

%Calculate Kernel
myRidge = 1e-8;
myEps = 1e-6;
myEpsLibsvm = 1e-4;
myEpsSMO = 1e-4;
% READ OPTIONAL PARAMETERS
args = varargin;
nargs = length(args);
for i=1:2:nargs
    switch args{i},
		case 'myEpsSMO',   myEpsSMO = args{i+1}; myEpsLibsvm=myEpsSMO;
    end
end

tic;
if (p > 1)
    if (rho ~= 2)
        rhobar = rho/(2-rho);% MKL norm in primal
        rhobarstar = rhobar/(rhobar-1);
    else
        rhobar = inf;% MKL norm in primal
        rhobarstar = 1;
    end
    ITER_MAX = 200;

%     % WARM START
%     eta = eta+min(.05,max(eta/2));
    eta = eta + min(eta(eta>0))/10;
    eta = eta/sum(eta);%     eta = ones(p,1)/p;

    % MIRROR DESCENT
    dualObj = 1.7977e+300;
    min_alpha = nan;
    min_b = nan;
    min_eta = nan;
    min_zeta = nan;
    min_dualObj = 1.7977e+300;
    min_delta = nan;
    min_aTQa_smo = nan;
    
    for iter_gamma = 1:ITER_MAX
        % Solve for alpha ;given eta
        dualObjOld = dualObj;
        if (sum(isnan(eta)))
            fprintf('[solve_ghkl_hinge_smo_rel] eta is NAN\n');
            keyboard;
        end
        fprintf('.');
        zetaFake = calculateZetaFake(affs,rho,weights,eta);
        zeta = zetaFake.^(1/rhobarstar);
        if (rho ~= 2)
            tic
            fprintf('@');
%             C = 1/lambda;L_p = rhobar;
            if iter_gamma == 1
                [alpha_smo,dualObj_smo,aTQa_smo] = smo_lpMKL_oneTask_noK_noCache(X,zeta,Y,abs(alpha_smo),1/lambda,rhobar,details,yKy,'myEpsSMO',myEpsSMO);
            else
                [alpha_smo,dualObj_smo,aTQa_smo] = smo_lpMKL_oneTask_noK_noCache(X,zeta,Y,alpha_smo,1/lambda,rhobar,details,yKy,'myEpsSMO',myEpsSMO);
            end
            bOpt = -1;
            fprintf('@');
            toc
            alphaOpt = alpha_smo.*Y;
            dualObj = dualObj_smo;
            fprintf('\nDualObj: %f\n',dualObj);
            if (dualObj < 0)
                fprintf('[solve_ghkl_hinge_smo_rel] Dual Obj is negative\n');
                keyboard;
            end

        else
            % USE LIBVSM - SOLVE L2-SVM DUAL
            Xzeta = bsxfun(@times,sqrt(zeta)',X);
            K2 = Xzeta*Xzeta' + eye(n)*(myRidge*sum(zeta));
            fprintf('&');
            svm_opt = sprintf('-q -t 4 -c %f -w1 %f -w-1 %f -e %f', 1/lambda, w2/n, w1/n, myEpsLibsvm);
            model = svmtrain(Y, [(1:n)',K2], svm_opt);
            fprintf('&');
            alphaOpt = zeros(n,1);
            alphaOpt(model.SVs) = abs(model.sv_coef);
            alphaOpt = alphaOpt.*Y;
            bOpt = -model.rho;
            if (model.Label(1) == -1)
                bOpt = -bOpt;
            end
            dualObj = (sum(abs(alphaOpt)) - .5*alphaOpt'*K2*alphaOpt);
            if (dualObj < 0)
                fprintf('[solve_ghkl_hinge_smo_rel] Dual Obj is negative\n');
                keyboard;
            end
            XYalpha = alphaOpt'*Xzeta;
            aTQa_smo = (XYalpha.^2)' + myRidge*(alphaOpt'*alphaOpt)*zeta;
        end
        
        delta = 2*(sum(abs(alphaOpt)) - dualObj);
        if (delta <= 0)
            fprintf('[solve_ghkl_hinge_smo_rel] Delta is negative\n');
            keyboard;
        end
        if ( 1e-6 <= (dualObj-dualObjOld)/dualObj)
            fprintf('[solve_ghkl_hinge_smo_rel] This is bad! dualObjOld=%e, dualObj=%e, rel.gap=%e\n',dualObjOld, dualObj,(dualObj-dualObjOld)/dualObj);
        end
        %book-keeping
        if (dualObj < min_dualObj)
            min_alpha = alphaOpt;
            min_b = bOpt;
            min_eta = eta;
            min_zeta = zeta;
            min_dualObj = dualObj;
            min_delta = delta;
            min_aTQa_smo = aTQa_smo;
        end
        %book-keeping
        if (abs((dualObjOld-dualObj)/dualObj) < myEps)
            break;
        end
        T2 = delta^rhobarstar;
        [answer flag] = calculateSumForDesc(affs,rho,weights,eta,aTQa_smo,zetaFake);
        if flag
            fprintf('[solve_ghkl_hinge_smo_rel] flag of [calculateSumForDesc] is true\n');
            keyboard;
        end
        gradF = -(delta/T2)/(2*rhobarstar)*answer;
        step = sqrt(log(p))/(sqrt(iter_gamma)*norm(gradF,inf));
        pvec = 1+log(eta) - step*gradF;
        eta = exp(pvec);
        eta = eta/sum(eta);
    end
    if (iter_gamma == ITER_MAX)
        fprintf('[solve_ghkl_hinge_smo_rel] Before convergence termination of MD, diff = %e\n',abs((dualObjOld-dualObj)/dualObj));
    end
    alphaOpt = min_alpha;
    bOpt = min_b;
    eta  = min_eta;
    zeta = min_zeta;
    dualObj = min_dualObj;
    delta = min_delta;
    aTQa_smo = min_aTQa_smo;
    
    finalLpDualObj = dualObj;
    % Theta Computation
    theta = (aTQa_smo/delta).^(1/(rhobar-1));
    omegaSquared = delta;
end

% % for p==1 use CVX
% if (p == 1)
%     [alphaOpt,bOpt,eta,omegaSquared,finalLpDualObj,theta,zeta,aTKa] = cvxLpNormSolver(eta,data,Y,lambda,rho,K);
% end

fprintf('[solve_ghkl_hinge_smo_rel] MDiterations: %d, MDtime: ',iter_gamma);
toc;

omegaSquared = omegaSquared*lambda/2;
finalLpDualObj = finalLpDualObj*lambda;
fprintf('[solve_ghkl_hinge_smo_rel] Objective = %e\n',finalLpDualObj);

%%%END function

function [zetaFake] = calculateZetaFake(affs,rho,weights,eta)
p =length(eta);
mat = zeros(p,p);
for i=1:p
    mat(i,affs{i}) = 1;
end
dffs = cell(p,1);
for i=1:p
    dffs{i} = find(mat(:,i)==1);
end

weightRho = weights.^(rho/(1-rho));
zetaFake = zeros(p,1);
etaFake = weightRho.*eta;
for i=1:p
    zetaFake(i,1) = norm(etaFake(dffs{i}),1-rho);
end


function [answer flag] = calculateSumForDesc(affs,rho,weights,eta,aTQa_smo,zetaFake)

if (rho ~= 2)
    rhobar = rho/(2-rho);% MKL norm in primal
    rhobarstar = rhobar/(rhobar-1);
else
    rhobar = inf;% MKL norm in primal
    rhobarstar = 1;
end
p =length(eta);

% etaRho = eta.^rho;
% 
% if all(etaRho)
%     weightRho = weights.^rho;
%     % aTKa = zeros(p,1);
%     % for i=1:p
%     %     aTKa(i) = alphaOpt'*K(:,:,i)*alphaOpt;
%     % end
%     aTKaZeta = (aTQa_smo.^(rhobarstar)).*(zetaFake.^(rho-1));
% 
%     answer = zeros(p,1);
%     for i=1:p
%         answer(i) = sum(aTKaZeta(affs{i}));
%     end
% 
%     flag = 0;
%     for i=1:p
% %         if (etaRho(i) > 0)
%             answer(i) = answer(i)*weightRho(i)/etaRho(i);
% %         else
% %             fprintf('[solve_ghkl_hinge_smo_rel] ethRho IS ZERO\n');
% %             flag = 1;
% %             keyboard;
% %         end
%     end
% else
    
    aTQa_smo = aTQa_smo./(zetaFake.^(1/rhobarstar));% aTQa_smo passed to the function has zetaFake in it.
    % aTKa = zeros(p,1);
    % for i=1:p
    %     aTKa(i) = alphaOpt'*K(:,:,i)*alphaOpt;
    % end
    aTKaZeta = (aTQa_smo.^(0.5/(rho-1))).*(zetaFake);

    answer = zeros(p,1);
    for i=1:p
        answer(i) = sum( (aTKaZeta(affs{i})*weights(i)/eta(i)).^rho );
    end
    flag = 0;
    if any( isnan(answer) | isinf(answer) )
        flag = 1;
        fprintf('[calculateSumForDesc] eta is zero\n');
        keyboard;
    end
    
% end
% fprintf('[calculateSumForDesc] keyboard\n');
% keyboard;



% % % solve the milasso for a given regularization parameter
% % n = size(data.X,1);
% % p =length(data.affinity)
% % pX = size(data.X,2);
% % affs = data.affinity;
% % weights = data.weights;
% %
% % X = data.X;	% no need to center here for hinge
% %
% % %Pratik
% % %     rhonorm = 1.5;
% %     Y1 = Y;
% %
% %     cvx_begin
% %         cvx_quiet(true);
% %         cvx_precision([.0001 .0001 .0001]);
% %         cvx_solver sdpt3
% %
% %         variable fw1(p);
% %         variable epsilon1(n);
% %         variable b;
% %         variable t1;
% %         expression fwDv1(p);
% %
% %         for i=1:p
% %             fwDv1(i) = norm(fw1(affs{i}),rhonorm);
% %         end
% %
% %         minimize 0.5*lambda*square(t1) + sum(epsilon1)
% %         subject to
% %             epsilon1 >= 1-(X*fw1+b).*Y1;
% %             epsilon1 >= 0;
% %             sum(fwDv1.*weights) <= t1;
% %
% %     cvx_end
% %
% %     if isnan(cvx_optval)
% %         fprintf('CVX GIVES NAN\n');
% %         keyboard;
% %     end
% %     if ~strcmp(cvx_status,'Solved')
% %         fprintf(strcat(cvx_status,'\n'));
% %     end
% %     b1 = b;
% %     omega1 = t1;
% %     x1 = weights.*fwDv1/omega1;
% %     eta1 = x1./weights.^2;
% %     lambdaVar = cell(1,p);
% %     a1 = abs(fw1);
% %     for i=1:p
% %         lambdaVar{i}=ones(length(affs{i}),1);
% %         if (rhonorm ~= 2)
% %             denominator = nthroot(norm(a1(affs{i}),rhonorm/2),2/(2-rhonorm));%fwDv1(i)^(2-rhonorm);
% %             for j=1:length(affs{i})
% %                 lambdaVar{i} = nthroot(a1(affs{i}),2/(2-rhonorm))/denominator;
% %             end
% %         end
% %     end
% %     zeta1 = zeros(p,1);
% %     for i=1:p
% %         zeta1(affs{i}) = zeta1(affs{i}) + 1./(eta1(i).*lambdaVar{i});
% %     end
% %     zeta1 = 1./zeta1;
% %     kappa1 = zeros(p,p);
% %     for i=1:p
% %         kappa1(i,affs{i}) = (zeta1(affs{i})./lambdaVar{i})/eta1(i);
% %     end
% %     %SVM for alpha
% %     %kernel computation
% %     K1 = zeros(n,n);
% %     for i=1:p
% %         K1 = K1+zeta1(i)*(X(:,i)*X(:,i)');
% %     end
% %     ridge = 1e-8;
% %     K1 = K1 + diag(ridge);
% %     K2 = [(1:n)',K1];
% %     addpath('~/libsvm-mat-2.9-1');
% % %     npos = sum(Y1==1);n;
% % %     svm_opt = sprintf('-t 4 -c %f -w1 %f -w-1 %f', 1/lambda, (n-npos)/n, npos/n);
% %     svm_opt = sprintf('-q -t 4 -c %f', 1/lambda);
% %     model = svmtrain(Y1, K2, svm_opt);
% %     alpha1 = zeros(n,1);
% %     alpha1(model.SVs) = abs(model.sv_coef);
% %     alpha1 = alpha1.*Y1;
% %     omegaSquared = t1*t1;
% %     objectiveValue = cvx_optval
% %     dualObjective = (sum(alpha1.*Y1) - .5*alpha1'*K1*alpha1)*lambda
% % %     wt = zeta1.*(X'*alpha1);
% % %     bprimal = -model.rho;
% % %     if (model.Label(1) == -1)
% % %         wt = -wt;
% % %         bprimal = -bprimal;
% % %     end
% % %     for i=1:p
% % %         wt1(i,1) = norm(wt(affs{i}),rhonorm);
% % %     end
% % %     primalObjective = 0.5*lambda*square(sum(wt1.*weights)) + sum(max(0,1-Y1.*(X*wt+bprimal)))
% % %     primalDual = .5*lambda*sum((wt.^2)./zeta1)+sum(max(0,1-Y.*(X*wt+bprimal)))
% % %     keyboard;
% % %Pratik
% %
% %


    %     %%%% NOT USED/NEEDED%%%%
    % %     %Epsilon computation
    %     boundedSV = find(abs(alphaOpt) == 1/lambda);
    %     %     keyboard;
    %     epsilonErr = zeros(n,1);
    %     if ~isempty(boundedSV)
    %         for i=1:p
    %             for j = boundedSV'
    %                 Km = K1(:,:,i);
    %                 epsilonErr(j) = epsilonErr(j) + theta(i)*sum(alphaOpt.*Km(:,j));
    %             end
    %         end
    %         oneVec = zeros(n,1);oneVec(boundedSV) = 1;
    %         epsilonErr = oneVec-Y.*(epsilonErr+oneVec*bOpt);
    %         epsilonErr(find(epsilonErr < 0)) = 0;
    %         if (~isempty(find(epsilonErr < 0)))
    %             fprintf('hinge loss is -ve\n');
    %             %             keyboard;
    %         end
    %         ziErr = 2*(finalLpDualObj - sum(epsilonErr))/lambda;
    %     else
    %         ziErr = 2*(finalLpDualObj - 0)/lambda;
    %     end
    %     %%%% NOT USED/NEEDED%%%%

    
%     cvx_begin
%     cvx_quiet(true);
%     cvx_precision([.0001 .0001 .0001]);
% %     cvx_solver sdpt3
%
%     variable alphaCVX(n)
%     variable t(p)
%     expression alphaYCVX(n)
%     expression aTKaCVX(p)
%
%     alphaYCVX = alphaCVX.*Y;
%     for i=1:p
%         aTKaCVX(i) = alphaYCVX'*K1(:,:,i)*alphaYCVX;
%     end
%
%     maximize sum(alphaCVX) - .5*norm(t,rhobarstar)
%
%     subject to
%     alphaCVX >= 0;
%     alphaCVX <= 1/lambda;
%     sum(alphaYCVX) >= 0;
%     sum(alphaYCVX) <= 0;
%     aTKaCVX <= t
% %     norm(aTKaCVX,rhobarstar) <= t;
%
%     cvx_end

