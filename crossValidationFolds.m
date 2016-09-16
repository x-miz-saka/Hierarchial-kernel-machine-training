function [folds compfolds] = crossValidationFolds(kfold,ntrain)

% cut into folds
folds = cell(1,kfold);
compfolds = cell(1,kfold);
for ifold = 1:kfold
    if ifold<kfold
        folds{ifold} = floor( (ifold-1)*ntrain/kfold + 1): floor( ifold*ntrain/kfold );
    else
        folds{ifold} = floor( (ifold-1)*ntrain/kfold + 1):ntrain;
    end
    compfolds{ifold}=1:ntrain;
    compfolds{ifold}(folds{ifold}) = [];
end