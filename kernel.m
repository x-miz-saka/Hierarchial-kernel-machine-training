function [k] = kernel(x, y)
% Quadratic and Cubic have their +1 factor removed (Pratik)
% function k = kernel(x, y);
%
%	x: (Lx,N) with Lx: number of points; N: dimension
%	y: (Ly,N) with Ly: number of points
%	k: (Lx,Ly)
%
%	KTYPE = 1:      linear kernel:      x*y'
%	KTYPE = 2,3,4:  polynomial kernel:  (x*y'*KSCALE+1)^KTYPE
%	KTYPE = 5:      sigmoidal kernel:   tanh(x*y'*KSCALE)
%	KTYPE = 6:	gaussian kernel with variance 1/(2*KSCALE)
%
%       assumes that x and y are in the range [-1:+1]/KSCALE (for KTYPE<6)

global KTYPE
global KSCALE

if(KTYPE ~= 6)
    k = x*y';
end
if KTYPE == 1				% linear
    % take as is
elseif KTYPE <= 4			% polynomial
    k = (k*KSCALE).^KTYPE; % changed - removed + 1 
elseif KTYPE == 5			% sigmoidal
    k = tanh(k*KSCALE);
elseif KTYPE == 6			% gaussian
%     KTYPE
    [Lx,N] = size(x);
%     Lx
    [Ly,N] = size(y);
%     Ly
% % % %     k = 2*k;
% % % %     k = k-sum(x.^2,2)*ones(1,Ly);
% % % %     k = k-ones(Lx,1)*sum(y.^2,2)';
% % % %     k = exp(k*KSCALE);
    k = ones(Lx,Ly);
    if (Lx ~= Ly)
        for i = 1:Lx
            for j = 1:Ly
                k(i,j) = exp(-KSCALE*norm(x(i,:)-y(j,:))^2);
            end
        end
    else
        for i = 1:Lx-1
            for j = i+1:Ly
                k(i,j) = exp(-KSCALE*norm(x(i,:)-y(j,:))^2);
                k(j,i) = k(i,j);
            end
        end
    end

%     size(k)
end