function  [alpha,dualObj,aTQa] = smo_lpMKL_oneTask_noK_noCache(XY,zeta,Y,alpha,C,L_p,details,yKy,varargin)

% lpmkl(K1(:,:,nonZero),ones(length(nonZero),1),1/lambda,mkl_norm,Y,w1/n,w2/n);

% T=1 (single task)
% XY - nxk matrix where n is number of training points and k is
% number of kernels. therefore: Q_i = YK_iY = YX_i*X_i'*Y*zeta_i = (XY)_i*(XY)_i'*zeta_i
% k is the number of kernels
% zeta - kx1 vector which needs to be multiplied to XY in order to get Q_i
% Y - nx1 vector of labels
% alpha - alpha of last iteration, nx1 zero vector by default
% C - regularization parameter
% L_p - p norm in the primal of lp-mkl
% details - details of the run. It contains
% details.balance 1 => apply proper weighing on C, 0 => otherwise
% details.numTrain - number of training instances
% details.wpos - weight of positive Xi
% details.wneg - weight of negative Xi
% yKy = 0 => X = YX (TODO - default), else if yKy = 1 => X = YX (nothing todo) - currently not used

display = 0;

myEps = 1e-4;% redundant 
myEps2 = 1e-3;
TAU = 1e-12;
DTAU = 1e-5;
obj_threshold = 0.01;
diff_threshold = 1e-5;
myRidge = 1e-8;
myTol = 1e-6;

% READ OPTIONAL PARAMETERS
args = varargin;
nargs = length(args);
for i=1:2:nargs
    switch args{i},
		case 'myEpsSMO',   myEps = args{i+1};
    end
end

k = size(XY,2);% number of kernels
n = details.numTrain;

wpos = details.wpos;
wneg = details.wneg;

% Norms
if (L_p == 1)
    L_q = inf;
else
    L_q = L_p/(L_p -1);
end

% Q = YK_trY
% if (yKy==0)
XY = bsxfun(@times,Y,XY);
% end
XY = bsxfun(@times,sqrt(zeta)',XY);
ridgeZeta = myRidge*zeta';

if (isempty(alpha))
    % cold start
    alpha = zeros(n,1);
    aTQa = zeros(k,1);
    alpha_sum = 0;
    T1 = 0;
    denom=0;
    numon = zeros(k,1);
    obj = 0;
    d = zeros(k,1);
    Qalpha = zeros(n,k);
else
    % warm start - for alpha & aTQa
    alpha_sum = sum(alpha);
    XYalpha = alpha'*XY;
    Qalpha = bsxfun(@times,XYalpha,XY) + alpha*ridgeZeta;
    aTQa = (alpha'*Qalpha)';
    T1 = norm(aTQa,L_q);
    denom = T1^(L_q - 1);
    numon = aTQa.^(L_q - 1);
    d = numon/denom;% kx1 vector
    obj = 0.5*T1;
    obj = obj - alpha_sum;
end

get_C = zeros(n,1);
get_C(Y==1) = C*wpos; %%%%%%%%%%% TO CHANGE
get_C(Y==-1) = C*wneg; %%%%%%%%%%% TO CHANGE

I_low = ((alpha>0)&(Y == 1)) | ((alpha<C*wneg)&(Y == -1)); % 1-0 vectors, one vector per task
I_up = ((alpha>0)&(Y == -1)) | ((alpha<C*wpos)&(Y == 1)); % 1-0 vectors, one vector per task

% SMO starts

% while KKT not satisfied
% choose the most violating pair
% update the pair

smoIter = 0;
nrIter = 0;
bisectIter = 0;
changeIter = 0;

while (true)
    smoIter = smoIter+1;
    if (mod(smoIter,1000)==0)
        fprintf('.');
    end
    
    % CALCULATE gradient w.r.t. alpha - G - nx1 vector
    % Update G : g_from_Qalpha();
    G = Qalpha*d-1;
    
    % select working set
    %checkKKT()
    iup = find(I_up);
    ilow = find(I_low);
    [maxVal max_i] = max( -(Y(iup)).*(G(iup)) );% max of -ve quantity = min of +ve quantity
    maxIndex = iup(max_i);
    ygrad_low = -(Y(ilow)).*(G(ilow));
    [minVal min_j] = min( ygrad_low );% min of -ve quantity = max of -ve quantity
    minIndex = ilow(min_j);
    gap = (maxVal - minVal);
    
    if (gap <= myEps)
        break;% SMO converged
    elseif (smoIter > 100000 && gap <= myEps2)
        break;
    end
    
    % main code
    % WSS1
    i = maxIndex;% I_up
    j = minIndex;% I_low
%     fprintf('%d, %d: ',i,j);
    % TODO WSS2
    
%     if (smoIter > 1)
%         if T1 <= 0
%             fprintf('[smo_lpMKL_singleTask] T1 is zero during WSS2 check\n');
%             keyboard;
%         end
%         t_t = ygrad_low < maxVal;
%         t = ilow(ygrad_low < maxVal);
%         s = -(Y(i)*Y(t));
%         b_it_sq = (maxVal - ygrad_low(t_t)).^2;
%         % calculating a_ij
%         h_ii_tt = ((XY([i;t],:)).^2)*d;%(squeeze((i,[i;t],:)))*d;% contains [H_ii H_it]' (1+t)x1 vector
%         h_it = bsxfun(@times,XY(i,:),XY(t,:))*d;%arrayfun(@(x)d'*squeeze(Q(x,x,:)),t,'UniformOutput',true);% contains H_tt, 1xt vector
%         d_aTQa = d./aTQa;
%         a_it = h_ii_tt(1) + h_ii_tt(2:end) + 2*s.*h_it + ...
%             2*(L_q-1)*( (bsxfun(@plus,bsxfun(@times,s,Qalpha(t,:)),Qalpha(i,:))).^2*d_aTQa - (( Qalpha(i,:)*d + s.*(Qalpha(t,:)*d) ).^2)/T1 );
%         [dummy t_j] = min(-b_it_sq./a_it);
%         j = t(t_j);
% %         fprintf('j: %d, minIndex: %d, wss2_j: %f, wss2_minIndex: %f, wss1_j: %f, wss1_minIndex: %f\n', j,minIndex,dummy,-b_it_sq(min_j)/a_it(min_j),...
% %             ygrad_low(t_j),ygrad_low(min_j));
%     end

    C_i = get_C(i);
    C_j = get_C(j);
    old_alpha_i=alpha(i);
    old_alpha_j=alpha(j);

    %Find a direction of descent 
    if Y(i)==Y(j)
        if G(i)-G(j) < G(j) - G(i)
            dir_i=1;
            dir_j=-1;
            dir_grad=G(i) - G(j);
            dir_grad_offset=0;%p[i] - p[j];
            eta_max=min(C_i - alpha(i), alpha(j));
        else
            dir_i=-1;
            dir_j=1;
            dir_grad=G(j) - G(i);
            dir_grad_offset=0;%p[j] - p[i];
            eta_max=min(alpha(i), C_j - alpha(j));
        end
        rr = (XY(i,:)-XY(j,:)).^2 + 2*ridgeZeta;
    else
        if G(i)+G(j) < -G(j) - G(i)
            dir_i=1;
            dir_j=1;
            dir_grad=G(i) + G(j);
            dir_grad_offset=-2;%p[i] + p[j];
            eta_max=min(C_i - alpha(i), C_j - alpha(j));
        else
            dir_i=-1;
            dir_j=-1;
            dir_grad=-G(i) - G(j);
            dir_grad_offset=2;%-p[j] - p[i];
            eta_max=min(alpha(i), alpha(j));
        end
        rr = (XY(i,:)+XY(j,:)).^2 + 2*ridgeZeta;
    end
    
    if dir_grad >= 0 || eta_max < 0
        fprintf('[smo_lpMKL_oneTask_noK_noCache] problem in dir_grad or eta_max\n');
        keyboard;
    end
    % Compute directional Hessian
    % Newton - raphson
    nrIter = nrIter+1;% eta = 0 (eta has been used instead of delta)

    % obtain dir_hess
    if (T1==0)
        dir_hess = 0;
    else
%         H1 = -(0.25*(dir_grad-dir_grad_offset)^2)/T1/L_p;
%         H1 = -(4*(dir_grad-dir_grad_offset)^2)/T1/L_p;
%         H1 = (4*(dir_grad-dir_grad_offset)^2)/T1/(L_q-1);
        H1 = (1-L_q)*(4*(dir_grad-dir_grad_offset)^2)/T1;
        TH1 = (dir_i*Qalpha(i,:) + dir_j*Qalpha(j,:));% b_k
        H2 = (2*rr*d + 4*(L_q-1)*(TH1.^2)*(d./aTQa));% 4 is due to square(2b_k)
        dir_hess = 0.5*(H1+H2);
%         if (dir_hess < 0) % 10/10/2011
%             fprintf('\n[smo_lpMKL_oneTask_noK_noCache] -ve hessian\n');
%             keyboard;
%         end
    end
    
    if dir_hess <= 0
        dir_hess=TAU;
    end
    
    old_obj = obj;
    old_dir_grad = dir_grad;
    old_eta = 0;
    eta_min = 0;
    eta = -dir_grad/dir_hess;
    if(eta > eta_max)
        eta = eta_max;
    elseif eta < eta_min
        eta = 0.5*(eta_max + eta_min);
    end
%     if display
%         fprintf('Obj: %f, dir_grad: %f\n',obj,dir_grad);
%         alpha'
%         fprintf('1: %f, 2: %f\n',aTQa(1),alpha'*Q(:,:,1)*alpha);
%     end
    while true
        new_alpha_i=old_alpha_i + eta*dir_i;
        new_alpha_j=old_alpha_j + eta*dir_j;
        change_i=new_alpha_i - alpha(i);
        change_j=new_alpha_j - alpha(j);
        alpha(i)=new_alpha_i;
        alpha(j)=new_alpha_j;
        
        % update_Qalpha - i.e. aTQa
        aTQa = aTQa + ((change_i*XY(i,:)+change_j*XY(j,:)).^2)' + myRidge*zeta*(change_i^2+change_j^2) + ...
            2*( Qalpha(i,:)*change_i+Qalpha(j,:)*change_j)';

        % update_Qalpha

%         for w=1:k
%             Qalpha(:,w) = Qalpha(:,w) + (change_i*XY(i,w)+change_j*XY(j,w))*XY(:,w);%Q(:,i,w)*change_i + Q(:,j,w)*change_j;
%         end
        
        Qalpha = Qalpha + bsxfun(@times,(change_i*XY(i,:)+change_j*XY(j,:)),XY);
%         
%         keyboard;
        Qalpha(i,:) = Qalpha(i,:) + ridgeZeta*change_i;
        Qalpha(j,:) = Qalpha(j,:) + ridgeZeta*change_j;
        
        alpha_sum = alpha_sum + change_i + change_j;
        T1 = norm(aTQa,L_q);
        denom = T1^(L_q - 1);
        numon = aTQa.^(L_q - 1);
        d = numon/denom;% kx1 vector
        obj = 0.5*T1;
        obj = obj - alpha_sum;
        
        % Compute new directional gradient
        dir_grad = dir_grad_offset;
        TH1 = (dir_i*Qalpha(i,:) + dir_j*Qalpha(j,:));% b_k
        largeOnes = d>DTAU;
        dir_grad = dir_grad + TH1(largeOnes)*d(largeOnes);
%         if display
%             fprintf('Obj: %f, change_i: %f, change_j: %f, dir_grad: %f\n',obj,change_i,change_j,dir_grad);
% %             alpha'
% %             fprintf('1: %f, 2: %f\n',aTQa(1),alpha'*Q(:,:,1)*alpha);
%         end
        if ( abs(eta - old_eta) < diff_threshold ||...
                abs(eta_max - eta_min) < diff_threshold ||...
                abs(dir_grad) < diff_threshold ||...
                (eta == eta_max && dir_grad < 0) ||...
                (eta == eta_min && dir_grad > 0) ||...
                (obj < old_obj + obj_threshold*eta*old_dir_grad) )% ||...
            % Small problem convergence
            break;
        end
        if(dir_grad > 0)
            eta_max=eta;
        elseif(eta >= eta_min+TAU)
            eta_min=eta;
        end
        
        % Try a Newton-Raphson step
        % Compute directional Hessian
        
        % obtain dir_hess
%         H1 = -(4*(dir_grad-dir_grad_offset)^2)/T1/L_p;
%         H1 = (4*(dir_grad-dir_grad_offset)^2)/T1/(L_q-1);
        H1 = (1-L_q)*(4*(dir_grad-dir_grad_offset)^2)/T1;
        H2 = (2*rr*d + 4*(L_q-1)*(TH1.^2)*(d./aTQa));% 4 is due to square(2b_k)
        dir_hess = 0.5*(H1+H2);

        if(dir_hess <= 0)
            dir_hess=TAU;
        end
        old_eta = eta;
        eta=eta - dir_grad/dir_hess;
        % If the result is outside the brackets, use bisection
        if(eta > eta_max || eta < eta_min)
            bisectIter = bisectIter+1;
            eta = 0.5*(eta_max + eta_min);
        end
        nrIter = nrIter+1;
    end

    if (display==1)
        fprintf('[smo_lpMKL_oneTask_noK_noCache] Value of delta: %f\n',eta);
    end
    
    flag_i=false;flag_j=false;
    if (alpha(i) < myTol)
        alpha(i) = 0;
        change_i = -alpha(i);
        flag_i = true;
    elseif (C_i - alpha(i) < myTol)
        alpha(i) = C_i;
        change_i = C_i-alpha(i);
        flag_i = true;
    end
    if (alpha(j) < myTol)
        alpha(j) = 0;
        change_j = -alpha(j);
        flag_j = true;
    elseif (C_j - alpha(j) < myTol)
        alpha(j) = C_j;
        change_j = C_j-alpha(j);
        flag_j = true;
    end
    if (flag_i && flag_j)
        aTQa = aTQa + ((change_i*XY(i,:)+change_j*XY(j,:)).^2)' + myRidge*zeta*(change_i^2+change_j^2) + ...
            2*( Qalpha(i,:)*change_i+Qalpha(j,:)*change_j)';
        Qalpha = Qalpha + bsxfun(@times,(change_i*XY(i,:)+change_j*XY(j,:)),XY);
        Qalpha(i,:) = Qalpha(i,:) + ridgeZeta*change_i;
        Qalpha(j,:) = Qalpha(j,:) + ridgeZeta*change_j;
        changeIter = changeIter+1;
    elseif flag_i
        aTQa = aTQa + ((change_i*XY(i,:)).^2)' + myRidge*zeta*(change_i^2) + ...
            2*( Qalpha(i,:)*change_i)';
        Qalpha = Qalpha + bsxfun(@times,(change_i*XY(i,:)),XY);
        Qalpha(i,:) = Qalpha(i,:) + ridgeZeta*change_i;
        changeIter = changeIter+1;
    elseif flag_j
        aTQa = aTQa + ((change_j*XY(j,:)).^2)' + myRidge*zeta*(change_j^2) + ...
            2*( Qalpha(j,:)*change_j)';
        Qalpha = Qalpha + bsxfun(@times,(change_j*XY(j,:)),XY);
        Qalpha(j,:) = Qalpha(j,:) + ridgeZeta*change_j;
        changeIter = changeIter+1;
    end
    
    % update I_up I_low accordingly
    if (alpha(i)<C_i && Y(i)==1) || (alpha(i)>0 && Y(i)==-1)
        I_up(i) = 1;
    else
        I_up(i) = 0;
    end
    if (alpha(i)<C_i && Y(i)==-1) || (alpha(i)>0 && Y(i)==1)
        I_low(i) = 1;
    else
        I_low(i) = 0;
    end
    if (alpha(j)<C_j && Y(j)==1) || (alpha(j)>0 && Y(j)==-1)
        I_up(j) = 1;
    else
        I_up(j) = 0;
    end
    if (alpha(j)<C_j && Y(j)==-1) || (alpha(j)>0 && Y(j)==1)
        I_low(j) = 1;
    else
        I_low(j) = 0;
    end
end
fprintf('[smo_lpMKL_oneTask_noK_noCache] gap: %e, smoIter: %d, nr: %d, bs: %d, change: %d\n',gap,smoIter,nrIter,bisectIter,changeIter);
% calculate final dual objective
dualObj = -obj;

%     % update the pair of alpha
%     i3 = i1;
%     if Y{t_v}(i3) == 1
%         C_i3 = C*wpos(t_v);
%     else
%         C_i3 = C*wneg(t_v);
%     end
%     if (alpha_t{t_v}(i3)+delta < myTol)
%         delta = -alpha_t{t_v}(i3);
%         %         x = delta;
%         alpha_t{t_v}(i3) = 0;
%     elseif (C_i3-alpha_t{t_v}(i3)-delta < myTol)
%         delta = C_i3-alpha_t{t_v}(i3);
%         %         x = delta;
%         alpha_t{t_v}(i3) = C_i3;
%     else
%         alpha_t{t_v}(i3) = alpha_t{t_v}(i3)+delta;
%     end
%     
%     i3 = i2;
%     if Y{t_v}(i3) == 1
%         C_i3 = C*wpos(t_v);
%     else
%         C_i3 = C*wneg(t_v);
%     end
%     if (alpha_t{t_v}(i3)+s*delta < myTol)
%         delta = -alpha_t{t_v}(i3)/s;
%         %         x = delta;
%         alpha_t{t_v}(i3) = 0;
%     elseif (C_i3-alpha_t{t_v}(i3)-s*delta < myTol)
%         delta = (C_i3-alpha_t{t_v}(i3))/s;
%         %         x = delta;
%         alpha_t{t_v}(i3) = C_i3;
%     else
%         alpha_t{t_v}(i3) = alpha_t{t_v}(i3)+s*delta;
%     end
    


% % calculate b_t
% b_t = zeros(T,1);
% % for t=1:T
% %     b_t(t) = ( (Y{t}(maxIndex(t))).*(grad{t}(maxIndex(t))) + (Y{t}(minIndex(t))).*(grad{t}(minIndex(t))) )/2;
% %     %     alpha_t{t} = alpha_t{t}.*Y{t};% alpha_t : alpha*Y
% % end



% % for testing correctness
% for t=1:T
%     for j=1:k
%         TestaTKa(t,j) = alpha_t{t}'*K_tr{t,j}*alpha_t{t};
%     end
% end
% for w=1:p
%     presentTasks = find(hull(w,:)==2);
%     for j=1:k
%         TestaTKSa(w,j) = sum(TestaTKa(presentTasks,j),1);
%     end
% end
% for w=1:p
%     TestBw(w) = nthroot(zeta_av_rhobarstar(w),rhobarstar)*norm(TestaTKSa(w,:),q);
% end
% for t=1:T
%     TestalphaSum(t) = sum(alpha_t{t});
% end
% TestdualObj = -0.5*norm(TestBw,rhobarstar) + sum(TestalphaSum);
%

% function [d obj] = calObj(aTQa,L_q)
%     obj = norm(aTQa,L_q);
%     denom = obj^(L_q - 1);
%     d = (aTQa.^(L_q - 1))/denom;% kx1 vector
%     obj = 0.5*obj;
