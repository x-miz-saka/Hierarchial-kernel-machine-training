
% for binary data
featureNum = 16;
conflictMatrix = zeros(featureNum);
for i=1:featureNum
    conflictMatrix(2*i-1,2*i)=1;
    conflictMatrix(2*i,2*i-1)=1;
end

% for tic tac toe
featureNum = 27;
conflictMatrix = zeros(featureNum);
conflictMatrix([2 3],1) = 1;
conflictMatrix([1 3],2) = 1;
conflictMatrix([1 2],3) = 1;
conflictMatrix([5 6],4) = 1;
conflictMatrix([4 6],5) = 1;
conflictMatrix([4 5],6) = 1;
conflictMatrix([8 9],7) = 1;
conflictMatrix([7 9],8) = 1;
conflictMatrix([7 8],9) = 1;
conflictMatrix([11 12],10) = 1;
conflictMatrix([10 12],11) = 1;
conflictMatrix([10 11],12) = 1;
conflictMatrix([14 15],13) = 1;
conflictMatrix([13 15],14) = 1;
conflictMatrix([13 14],15) = 1;
conflictMatrix([17 18],16) = 1;
conflictMatrix([16 18],17) = 1;
conflictMatrix([16 17],18) = 1;
conflictMatrix([20 21],19) = 1;
conflictMatrix([19 21],20) = 1;
conflictMatrix([19 20],21) = 1;
conflictMatrix([23 24],22) = 1;
conflictMatrix([22 24],23) = 1;
conflictMatrix([22 23],24) = 1;
conflictMatrix([26 27],25) = 1;
conflictMatrix([25 27],26) = 1;
conflictMatrix([25 26],27) = 1;