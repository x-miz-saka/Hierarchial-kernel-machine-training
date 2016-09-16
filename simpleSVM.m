function [alphaOpt, bOpt, dualObj] = simpleSVM(data,Y,lambda,theta,zeta)

n = size(data.X,1);
p =length(data.affinity);
X = data.X;
w1 = sum(Y == 1);
w2 = sum(Y == -1);
if (w1+w2 ~= n)
    'Problem IN SMALL PROBLEM SOLVER';
    keyboard;
end
%Calculate Kernel
myRidge = 1e-8;
myEps = 1e-6;%1e-12;

K1 = zeros(n,n);
for i=1:p
    fac = zeta(i)*theta(i);
    if (fac > 0)
        K1 = K1 + (X(:,i)*X(:,i)'+eye(n)*myRidge)*fac;
    else if (fac < 0)
            fprintf('\nsimpleSVM: FAC is -ve\n');
            keyboard;
        end
    end
end
fprintf('\n@');
svm_opt = sprintf('-q -t 4 -c %f -w1 %f -w-1 %f -e %f', 1/lambda, w2/n, w1/n, myEps);
model = svmtrain(Y, [(1:n)',K1], svm_opt);
fprintf('@\n');
alphaOpt = zeros(n,1);
alphaOpt(model.SVs) = abs(model.sv_coef);
alphaOpt = alphaOpt.*Y;
bOpt = -model.rho;
if (model.Label(1) == -1)
    bOpt = -bOpt;
end
dualObj = (sum(abs(alphaOpt)) - .5*alphaOpt'*K1*alphaOpt);
dualObj = dualObj*lambda;
if (dualObj < 0)
    fprintf('Dual Obj is negative\n');
    keyboard;
end
fprintf('\nObjective after alpha comptutation: %e\n', dualObj);
