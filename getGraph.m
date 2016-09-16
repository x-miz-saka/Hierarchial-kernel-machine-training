function graphModel = getGraph(hull)

p = size(hull,2);
n = size(hull,1);
graphModel = cell(n,1);
for i=1:n
    graphModel{i} = find(hull(i,:)==2);
end

% %%%PRINTING
% for i=1:n
%     fprintf('\n Feature parents of Node: %d: ',i);
%     gm = graphModel{i};
%     for j=1:length(gm)
%         fprintf('%d ', gm(j));
%     end
%     fprintf('\n');
% end
% %%%PRINTING
