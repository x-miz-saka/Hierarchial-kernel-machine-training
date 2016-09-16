function [activeHull] = indexActiveHull(active, affs)

p =length(affs);
% store descendents
mat = zeros(p,p);
for i=1:p
    mat(i,affs{i}) = 1;
end

% get ancestors
dffs = cell(p,1);
for i=1:p
    dffs{i} = find(mat(:,i)==1);
end

%get hull
activeHull = zeros(p,1);
for i = active' %ROW vector should be given
    activeHull(dffs{i}) = 1;
end

activeHull = find(activeHull);
if (activeHull(1)~=1)
    activeHull = [1;activeHull];
end
% keyboard;