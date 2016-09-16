function P = perms_reps(vec,reps)
% permutations with replicates, done recursively
%
% arguments:
% vec - vector of elements
% reps - number of replicates of each elements allowed
%
% P - array of permutations as rows of P
%
% Example usage:
% Permutations of the vector [1 1 2 2 3]
% vec = [1 2 3];
% reps = [2 2 1];
% P = perms_reps(vec,reps);
 
% total number of elements
ne = sum(reps);
n = length(vec);
 
% special cases
if ne == 0
  P = [];
elseif ne == 1
  k = find(reps);
  P = vec(k);
else
  % there are at least two elements
  P = [];
  for i = 1:n
    if reps(i)>0
      repst=reps;
      repst(i)=repst(i)-1;
      P_i = perms_reps(vec,repst);
      ni = size(P_i,1);
      P = [P;[repmat(vec(i),ni,1),P_i]];
    end
  end
end
