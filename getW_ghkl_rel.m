function getW_ghkl_rel(outputsRho,modelRho)

p = size(outputsRho{1}.theta,1);
w = zeros(p,1);
for i=1:p
    Xloctrain = get_data_reduced(outputsRho{1}.hull(i,:),modelRho.data);
    w(i) = outputsRho{1}.theta(i)*sqrt(outputsRho{1}.zeta(i))*sum(Xloctrain.*outputsRho{1}.alpha);
end