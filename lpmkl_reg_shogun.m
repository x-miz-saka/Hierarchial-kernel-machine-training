function [alpha,b] = lpmkl_reg_shogun(K,kernel_weights,C,mkl_norm,Y)

% addpath('');
myEps = 1e-4;
tube_epsilon = .01;
p = size(K,3);
if isempty(kernel_weights)
    kernel_weights = ones(p,1);
end
sg('new_classifier', 'MKL_REGRESSION');                % using multiclass MKL
sg('mkl_use_interleaved_optimization', 1); % 0, 1
% 'AUTO'
sg('set_solver', 'AUTO'); % DIRECT, NEWTON, CPLEX, AUTO, GLPK, ELASTICNET
sg('clean_kernel');                                    % preparing for Shogun
sg('clean_features','TRAIN');                          %

% size_cache = 5;
% sg('set_kernel', 'CUSTOM', 0);              % using Multiple kernel

for m=1:p
    sg('add_kernel', kernel_weights(m),'CUSTOM', K(:,:,m), 'FULL');     % in shogun-0.9.3
end

sg('set_labels', 'TRAIN', Y');            % set labels for training

% Parameters for MKL
epsilon = myEps;
% C = 1;
mkl_eps = myEps;
% mkl_norm = 2;                                  % set Lp_norm (p=2)
sg('svm_epsilon', epsilon);                    % set epsilon for
sg('c',C);                                    % set C parameter for SVM 
sg('svr_tube_epsilon', tube_epsilon);
sg('mkl_parameters', mkl_eps, 0, mkl_norm);    % set mkl_parameters

% MKL Training
sg('train_regression');                        % train MKL classifiers
[b,alpha]=sg('get_svm');
% [objective] = sg('compute_mkl_dual_objective');
% [objective1] = sg('compute_mkl_primal_objective');
% objective1
% sg('compute_mkl_primal_objective')

% % Testing phase
% sg('clean_kernel');                            % Preparing to set kernels for testing phase
% sg('set_kernel', 'COMBINED', size_cache);
% for m=1:size(Kt,3)
%     sg('add_kernel', 1,'CUSTOM', Kt(:,:,m),'FULL');     % Test shogun-0.9.3
% end
% 
% result=sg('classify');                                  % Classify samples

