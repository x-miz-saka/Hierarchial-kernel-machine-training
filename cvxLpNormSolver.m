function [alphaOpt,bOpt,eta,omegaSquared,finalLpDualObj,theta,zeta,aTKa] = cvxLpNormSolver(eta,data,Y,lambda,rho,K)

n = size(data.X,1);
p =length(data.affinity);
affs = data.affinity;
weights = data.weights;
% fprintf('\nNo. of nodes: %d\n',p);
X = data.X;	% no need to center here for logistic
w1 = sum(Y == 1);
w2 = sum(Y == -1);
if (w1+w2 ~= n)
    'Problem IN SMALL PROBLEM SOLVER';
    keyboard;
end


rhonorm = rho;
Y1 = Y;
pos = find(Y==1);
neg = find(Y== -1);
cvx_begin
%     cvx_quiet(true);
    %     cvx_precision([.0001 .0001 .0001]);
    cvx_solver sdpt3

    variable fw1(p);
    variable epsilon1pos(w1);
    variable epsilon1neg(w2);
    variable b;
    variable t1;
    expression fwDv1(p);

    for i=1:p
        fwDv1(i) = norm(fw1(affs{i}),rhonorm);
    end

    %     minimize 0.5*square(t1) + sum(epsilon1)/lambda
    minimize 0.5*square(t1) + (w2/n)*sum(epsilon1pos)/lambda + (w1/n)*sum(epsilon1neg)/lambda
    subject to
    epsilon1pos >= 1-(X(pos,:)*fw1+b).*Y1(pos);
    epsilon1neg >= 1-(X(neg,:)*fw1+b).*Y1(neg);
    epsilon1pos >= 0;
    epsilon1neg >= 0;
    sum(fwDv1.*weights) <= t1;

cvx_end

if isnan(cvx_optval)
    fprintf('CVX GIVES NAN\n');
    keyboard;
end
if ~strcmp(cvx_status,'Solved')
    fprintf(strcat(cvx_status,'\n'));
end
if (strcmp(cvx_status,'Solved'))
    b1 = b;
    omega1 = t1;
    x1 = weights.*fwDv1/omega1;
    eta1 = x1./weights.^2;
    lambdaVar = cell(1,p);
    a1 = abs(fw1);
    for i=1:p
        lambdaVar{i}=ones(length(affs{i}),1);
        if (rhonorm ~= 2)
            denominator = nthroot(norm(a1(affs{i}),rhonorm/2),2/(2-rhonorm));%fwDv1(i)^(2-rhonorm);
            for j=1:length(affs{i})
                lambdaVar{i} = nthroot(a1(affs{i}),2/(2-rhonorm))/denominator;
            end
        end
    end
    zeta1 = zeros(p,1);
    for i=1:p
        zeta1(affs{i}) = zeta1(affs{i}) + 1./(eta1(i).*lambdaVar{i});
    end
    zeta1 = 1./zeta1;
    
    K3 = zeros(n,n);
    for i=1:p
        K3 = K3+zeta1(i)*K(:,:,i);%(X(:,i)*X(:,i)');
    end
    
    K2 = [(1:n)',K3];
    svm_opt = sprintf('-q -t 4 -c %f -w1 %f -w-1 %f', 1/lambda, w2/n, w1/n);
    model = svmtrain(Y1, K2, svm_opt);
    alpha1 = zeros(n,1);
    alpha1(model.SVs) = abs(model.sv_coef);
    alpha1 = alpha1.*Y1;
    currentValue = t1*t1;
    currentValue2 = alpha1'*K3*alpha1;
    objectiveValue = cvx_optval;
    dualObjective = (sum(abs(alpha1)) - .5*alpha1'*K3*alpha1);
    if (p > 1)
        fprintf('OBJECTIVE: cvx primal %f, dual %f\n', objectiveValue,dualObjective);%finalLpDualObj
        fprintf('CURRENT VALUE: cvx %f cvx_dual %f\n\n', currentValue, currentValue2);%delta
    end
    %     wt = zeta1.*(X'*alpha1);
    %     bprimal = -model.rho;
    %     if (model.Label(1) == -1)
    %         wt = -wt;
    %         bprimal = -bprimal;
    %     end
    %     for i=1:p
    %         wt1(i,1) = norm(wt(affs{i}),rhonorm);
    %     end
    %     primalObjective = 0.5*lambda*square(sum(wt1.*weights)) + sum(max(0,1-Y1.*(X*wt+bprimal)))
    %     primalDual = .5*lambda*sum((wt.^2)./zeta1)+sum(max(0,1-Y.*(X*wt+bprimal)))
    
    
    if (p >= 1)
        alphaOpt = alpha1;
        bOpt = b1;
        eta = eta1;
        omegaSquared = currentValue;
        zeta = 1;% DUMMY (NOT CORRECT VALUE)
        finalLpDualObj = cvx_optval;
        theta = 1;
        aTKa = 2*(sum(alpha1) - dualObjective);
        
    end
end



%Pratik

%     cvx_begin
%     cvx_quiet(true);
%     %     cvx_precision([.0001 .0001 .0001]);
%     cvx_solver sdpt3
%
%     variable fw1(p);
%     variable epsilon1(n);
%     variable b;
%     variable t1;
%     expression fwDv1(p);
%
%     for i=1:p
%         fwDv1(i) = norm(fw1(affs{i}),rhonorm);
%     end
%
%     minimize 0.5*square(t1) + sum(epsilon1)/lambda
%     subject to
%     epsilon1 >= 1-(X*fw1+b).*Y1;
%     epsilon1 >= 0;
%     sum(fwDv1.*weights) <= t1;
%
%     cvx_end
