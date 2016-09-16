function  [alpha,dualObj,aTQa] = smo_lpMKL_oneTask_K(K_tr,Y,alpha,C,L_p,details,yKy)

% lpmkl(K1(:,:,nonZero),ones(length(nonZero),1),1/lambda,mkl_norm,Y,w1/n,w2/n);

% T=1 (single task)
% K_tr - 3-d matrix of size nxnxk. where n is number of training points and
% k is the number of kernels must be PD % YKY will be computed here depending on yKy
% Y - nx1 vector of labels
% alpha - alpha of last iteration, nx1 zero vector by default
% C - regularization parameter
% L_p - p norm in the primal of lp-mkl
% details - details of the run. It contains
% details.balance 1 => apply proper weighing on C, 0 => otherwise
% details.numTrain - number of training instances
% details.wpos - weight of positive Xi
% details.wneg - weight of negative Xi
% yKy = 0 => Q = YK_trY (TODO), else if yKy = 1 => K_tr = YKY (nothing todo)

display = 0;

myEps = 1e-3;
TAU = 1e-12;
DTAU = 1e-5;
obj_threshold = 0.1;
diff_threshold = 1e-5;

k = size(K_tr,3);% number of kernels
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
if (yKy==0)
    indPos = Y==1;
    indNeg = Y==-1;
    YY = ones(n,n);
    YY(indPos,indNeg) = -1;
    YY(indNeg,indPos) = -1;
    Q = bsxfun(@times,YY,K_tr);
end

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
    aTQa = zeros(k,1);
    Qalpha = zeros(n,k);
    for w=1:k
%         aTQa(w) = alpha'*Q(:,:,w)*alpha;%vectorize_quad_single(Q(:,:,w),alpha);% can be bettered for sparse matrix
        Qalpha(:,w) = Q(:,:,w)*alpha;
        aTQa(w) = alpha'*Qalpha(:,w);
    end
%     [d normTerm] = calObj(aTQa,L_q);% d: kx1, normTerm: 1x1
    
    T1 = norm(aTQa,L_q);
    denom = T1^(L_q - 1);
    numon = aTQa.^(L_q - 1);
    d = numon/denom;% kx1 vector
    obj = 0.5*T1;
    obj = obj - alpha_sum;
    
%     Qalpha = shiftdim(alpha'*Q,1);% vector nxk
end

get_C = zeros(n,1);
get_C(Y==1) = C;C*wpos;
get_C(Y==-1) = C;C*wneg;

I_low = ((alpha>0)&(Y == 1)) | ((alpha<C*wneg)&(Y == -1)); % 1-0 vectors, one vector per task
I_up = ((alpha>0)&(Y == -1)) | ((alpha<C*wpos)&(Y == 1)); % 1-0 vectors, one vector per task

% SMO starts

% while KKT not satisfied
% choose the most violating pair
% update the pair

% i_old =0;j_old=0;

smoIter = 0;
nrIter = 0;
bisectIter = 0;

% newtonRaphsonIter = zeros(50,1);
% objVec = [];counter = 0;

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
        %         fprintf('smo-htl - SMO convergence\n')
        break;% SMO converged
    end
    
    % main code
    % WSS1
    i = maxIndex;% I_up
    j = minIndex;% I_low

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
%         h_i_it = (squeeze(Q(i,[i;t],:)))*d;% contains [H_ii H_it]' (1+t)x1 vector
%         h_t_t = arrayfun(@(x)d'*squeeze(Q(x,x,:)),t,'UniformOutput',true);% contains H_tt, 1xt vector
%         d_aTQa = d./aTQa;
%         a_it = h_i_it(1) + h_i_it(2:end) + 2*s.*h_t_t + ...
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
        rr=(squeeze(Q(i,i,:) + Q(j,j,:) - 2*Q(i,j,:)))';
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
        rr=(squeeze(Q(i,i,:) + Q(j,j,:) + 2*Q(i,j,:)))';
    end
    
    if dir_grad >= 0 || eta_max < 0
        fprintf('[smo_lpMKL_oneTask_K] problem in dir_grad or eta_max\n');
        keyboard;
    end
    % Compute directional Hessian
    % Newton - raphson
    nrIter = nrIter+1;% eta = 0 (eta has been used instead of delta)

    % obtain dir_hess
    if (T1==0)
        dir_hess = 0;
    else
        H1 = (1-L_q)*(4*(dir_grad-dir_grad_offset)^2)/T1;
        TH1 = (dir_i*Qalpha(i,:) + dir_j*Qalpha(j,:));% b_k
        H2 = (2*rr*d + 4*(L_q-1)*(TH1.^2)*(d./aTQa));% 4 is due to square(2b_k)
        dir_hess = 0.5*(H1+H2);
%         if (dir_hess < 0) % 10/10/2011
%             fprintf('\n-ve hessian\n');
%             keyboard;
%         end
    end
    
    if dir_hess <= 0
        dir_hess=TAU;
    end
    
    old_obj = obj;
    old_dir_grad = dir_grad;
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
%         counter = counter+1;
%         objVec(counter) = obj;
        new_alpha_i=old_alpha_i + eta*dir_i;
        new_alpha_j=old_alpha_j + eta*dir_j;
        change_i=new_alpha_i - alpha(i);
        change_j=new_alpha_j - alpha(j);      
        alpha(i)=new_alpha_i;
        alpha(j)=new_alpha_j;
        
        % update_Qalpha - i.e. aTQa
        aTQa = aTQa + squeeze( Q(i,i,:)*change_i*change_i + Q(j,j,:)*change_j*change_j + 2*Q(i,j,:)*change_i*change_j ) +...
            2*( Qalpha(i,:)*change_i+Qalpha(j,:)*change_j)'; %a*delta^2 + 2*b*delta;
        
        % update_Qalpha
        for w=1:k
            Qalpha(:,w) = Qalpha(:,w) + Q(:,i,w)*change_i + Q(:,j,w)*change_j;
        end
        
        alpha_sum = alpha_sum + change_i + change_j;
%         [d normTerm] = calObj(aTQa,L_q);
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
        if ( obj < old_obj + obj_threshold*eta*old_dir_grad ||...
                abs(eta_max - eta_min) < diff_threshold ||...
                abs(dir_grad) < diff_threshold ||...
                (eta == eta_max && dir_grad < 0) ||...
                (eta == eta_min && dir_grad > 0) )
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
        H1 = (1-L_q)*(4*(dir_grad-dir_grad_offset)^2)/T1;
        H2 = (2*rr*d + 4*(L_q-1)*(TH1.^2)*(d./aTQa));% 4 is due to square(2b_k)
        dir_hess = 0.5*(H1+H2);

        if(dir_hess <= 0)
            dir_hess=TAU;
        end
        eta=eta - dir_grad/dir_hess;
        % If the result is outside the brackets, use bisection
        if(eta > eta_max || eta < eta_min)
            bisectIter = bisectIter+1;
            eta = 0.5*(eta_max + eta_min);
        end
        nrIter = nrIter+1;
    end
    
    % update gradients w.r.t. alpha using d, Qalpha
    
%     newtonRaphsonIter(smoIter,1) = nrIter;
    
    if (display==1)
        fprintf('[smo_lpMKL_singleTask] Value of delta: %f\n',eta);
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
    
end

% % calculate b_t
% b_t = zeros(T,1);
% % for t=1:T
% %     b_t(t) = ( (Y{t}(maxIndex(t))).*(grad{t}(maxIndex(t))) + (Y{t}(minIndex(t))).*(grad{t}(minIndex(t))) )/2;
% %     %     alpha_t{t} = alpha_t{t}.*Y{t};% alpha_t : alpha*Y
% % end

% calculate final objective
dualObj = -obj;

fprintf('[smo_lpMKL_oneTask_K] - gap: %f, smoIter: %d, nr: %d, bs: %d\n',gap,smoIter,nrIter,bisectIter);
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
