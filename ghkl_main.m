load('binaryDatasetExample.mat')%uncomment it for a trial run

%%%%%%PARAMETERS%%%%%%%%
weights = [1 .1 2];% 1st is dummy, second is the weight of TOP node, third is the parameter 'a' such that d_v=a^|v|
solver = 'smo_rel';%smo_rel,smo,shogun, for REL application use smo_rel since it is the fastest of the three
rhonorm = 1.1;%(1<rhonorm<=2)
loss = 'hinge';
kernel = 'polynomial';%for REL application use 'polynomial'
mingap = 1e-2; % relative duality gap for active set algorithm
maxactive = 10000;% maximum size of the active set
memory_cache = 1*1e9;% available memory in bytes (default=1e9(this has a significant impact on performance, i.e., do use your memory!))
high = 3;
low = -3;
%%%%%%PARAMETERS%%%%%%%%

% ADD LIBSVM PATH in case of rhonorm=2 (OR change the code to use the inbuilt SMO algorithm)
% LOAD DATA FILE containing feature matrix X and label vector Y
% X: n x d matrix where n is number of data points and d is dimension.
% Y is a n x 1  (for classification, a {-1,1}^n vector)
% For REL, X should be a {0,1} matrix

d = size(X,2);

% default value of conflictMatrix
conflictMatrix = zeros(d,d);

% Concept of Conflict Matrix is added to make the program run faster in
% case we have any prior knowledge about the features of the given dataset.
% It should be changed ONLY when code is run for Rule Ensemble Learning. If
% code is run only for Generalized HKL (i.e. not REL) it SHOULD be a zero matrix.

% Conflict matrix captures the information whether the basic propositions (feature_i==1) &
% (feature_j==1) can coexist in a rule. If yes, the conflictMatrix(i,j) = conflictMatrix(i,j) = 0;
% Else conflictMatrix(i,j) = conflictMatrix(i,j) = 1;
% In case of no prior knowledge, conflictMatrix should be a zero matrix
% which means all basic propositions can co-occur with each other.
% Example of a case where prior knowledge is present is when features are
% of complementary type: A, (NOT A). We know for sure that basic propositions (A==1) and (NOT A ==1)
% cannot coexist in a rule. Code ran approximately 20-30% faster when such prior
% knowledge was available.


range = high - low + 1;
lambdas = logspace(high,low,range);% 1/C, C: misclassification penalt

% Prepare the train-test split - XTrain, YTrain, XTest, YTest

[outputsErr,modelErr,accuraciesErr] = ghkl(XTrain,YTrain,lambdas,loss,kernel,weights,'Xtest',XTest,...
            'Ytest',YTest,'display',0,'maxactive',maxactive,'memory_cache',memory_cache, 'rhonorm',rhonorm,'conflictMatrix',conflictMatrix,...
            'solver',solver,'mingap',mingap);            
            
            
% In order to see the rules (for REL application) please use the following command:
% graphModel = getGraph(outputsRho{1}.hull(outputsRho{1}.theta > 0,:))
% graphModel will be a r x 1 cell where r is the number of rules
% each row of the cell will contain one rule (indices will represent the
% feature indices of feature matrix). For example: if graphModel = {[1 2 4], [3 4 5]};
% it means the rules are: 
% R1: feature_1 AND feature_2 AND feature_4
% R2: feature_3 AND feature_4 AND feature_5
% Whether these rules favour +ve or -ve class can be see by calculating the
% weight vectors calculated by getW_ghkl_rel.m
