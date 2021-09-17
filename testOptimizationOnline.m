clear;close all;clc

select=3;
datasetSelect;


pts = 100;
iterations = 100;
disp('RpropOnline');disp(pts)
outNum= size(Y_test,2);inNum= size(X_test,2);
sign = zeros(1,outNum);
sigf = zeros(1,outNum);
ls = zeros(inNum,outNum);
R = zeros(pts,outNum);
X= X_train; clear X_train;
Ya= Y_train; clear Y_train;
for j = 1:1%outNum
    disp(j)
    clear rpo
    tic;
    rpo = RpropOnline;
%     rpo.X = X_train(1:pts,:)'; rpo.Y = Y_train(1:pts,j)'; if offline
    for k =1 :500000
        rpo.X = [rpo.X,X(k,:)']; rpo.Y = [rpo.Y,Ya(k,j)']; %if online
        rpo.PR(); %optimize
        R(k,j) = rpo.lik;
        auxdelta(k,1) = ( rpo.delta(2));
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
