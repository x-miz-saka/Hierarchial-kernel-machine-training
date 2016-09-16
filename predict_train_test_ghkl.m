function [predtrain,predtest] = predict_train_test_ghkl(datatest,datatrain,hull,alpha,b,zeta,theta);

switch datatrain.dag_type
    case {'grid + input_space' , 'mkl + input_space', 'bimkl + input_space' }
        % prepare data
        p = size(theta,1);
        n = datatrain.n;
        m = datatest.n;
        predtrain = zeros(datatrain.n,1)+b;
        predtest = zeros(datatest.n,1)+b;

        myRidge = (1e-8);
%         outer = @(x)(x*x'+eye(length(x))*myRidge);
       
        theta_zeta = zeta.*theta;
        nonZero = find(theta_zeta);
%         predtrain_cell = arrayfun(@(x)((outer(get_data_reduced(hull(x,:),datatrain))*alpha)*theta_zeta(x)),nonZero,'UniformOutput', false);
        for i=nonZero'
            Xloctrain = get_data_reduced(hull(i,:),datatrain);
            Xloctest = get_data_reduced(hull(i,:),datatest);
            predtrain = predtrain + ( (Xloctrain*(Xloctrain'*alpha) + eye(n)*alpha*myRidge) )*theta_zeta(i);
            predtest = predtest + (Xloctest*(Xloctrain'*alpha))*theta_zeta(i);
        end
        
%         for i=1:p
%             Xloctrain = get_data_reduced(hull(i,:),datatrain);
%             KTrain(:,:,i) = (Xloctrain*Xloctrain'+eye(n)*myRidge)*zeta(i);
%         end
%         for i=1:p
%                 Km = KTrain(:,:,i);
%                 predtrain = predtrain + theta(i)*(Km*alpha);
%         end
        
%         KTest = zeros(m,n,p);
%         for i=1:p
%             Xloctrain = get_data_reduced(hull(i,:),datatrain);
%             Xloctest = get_data_reduced(hull(i,:),datatest);
%             KTest(:,:,i) = (Xloctest*Xloctrain')*zeta(i);
%         end
%         for i=1:p
%                 Km = KTest(:,:,i);
%                 predtest = predtest + theta(i)*(Km*alpha);
%         end



%         for i1=1:size(hull,1)
%             Xloctrain = get_data_reduced(hull(i1,:),datatrain);
%             Xloctest = get_data_reduced(hull(i1,:),datatest);
%             predtrain = predtrain + zeta(i1) * Xloctrain * ( Xloctrain' * alpha );
%             predtest = predtest + zeta(i1) * Xloctest * ( Xloctrain' * alpha );
%         end

end
