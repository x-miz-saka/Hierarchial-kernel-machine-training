function [fscore]=calculateFscore(Predicted,Actual)

Predicted = sign(Predicted);

FP = sum(double( (Predicted == 1) & (Actual == -1) ));
FN = sum(double( (Predicted == -1) & (Actual == 1) ));

TwoTP = 2 * sum(double( (Predicted == 1) & (Actual == 1) ));
fscorePos = TwoTP / (TwoTP + FP + FN);

TwoTN = 2 * sum(double( (Predicted == -1) & (Actual == -1) ));
fscoreNeg = TwoTN / (TwoTN + FP + FN);

fscore = .5*(fscorePos+fscoreNeg);

