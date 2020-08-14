clear;close all;clc
datasetSelect;
pts = 100;
iterations = 200;
disp('RpropOnline');disp(pts)
outNum= size(Y_test,2);inNum= size(X_test,2);
sign = zeros(1,outNum);
sigf = zeros(1,outNum);
ls = zeros(inNum,outNum);
R = zeros(pts,outNum);
for j = 1:outNum
    disp(j)
    clear rpo
    tic;
    rpo = RpropOnline;
    rpo.X = X_train(1:pts,:)'; rpo.Y = Y_train(1:pts,j)';
    for k =1 :iterations
%         rpo.X = [rpo.X,X(:,k)]; rpo.Y = [rpo.Y,Ya(j,k)]; if online
        rpo.PR(); %optimize
        R(k,j) = rpo.lik;
    end
    toc
        
    sigf(j) = rpo.sigmaF;
    sign(j) = rpo.sigmaN;
    ls(:,j)  = rpo.lengthS;
    
end
beep;
save(['C:\Users\alejandro\Desktop\P10\Datasets\hyps_',fileName,'.mat'],...
    'ls','sign','sigf');
%   save('hyps','ls','sign','sigf')
