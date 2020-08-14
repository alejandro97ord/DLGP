clear all;close all;clc
% opengl('save','software')
select = 6;
datasetSelect;
rng(0);

inputSize =size(X_train,2);
amountDoF = size(Y_train,2);

Nsteps = 1;
Ns = round([1,linspace(100,size(X_train,1),Nsteps)]);

output = zeros(1,size(X_test,1));
outs = [];

outvar = zeros(1,size(X_test,1));
negll = zeros(1,size(X_test,1));
Nll = zeros(amountDoF,100);
oVar= zeros(amountDoF,100);
t_update = zeros(amountDoF,100);
t_pred = zeros(amountDoF,100);
error = zeros(amountDoF,100);
% oEff = zeros(amountDoF,3);
start = strcat('Start time: ',datestr(now,'HH.MM.SS'));disp(start)
runTime = tic;
for DoF =1:amountDoF
   dataGeneration; 
   outs = [outs; output];
end
runTime = toc(runTime);
start = strcat('End time: ',datestr(now,'HH.MM.SS'));disp(start)
disp('     error       NLL')
disp([error(:,end), Nll(:,end)])
%%
saveFile = strcat('testResults/',fileName,datestr(now,'mmm.dd_HH.MM.SS'),'.mat');

specs.pts = gp01.pts; specs.N = gp01.N; specs.divM = gp01.divMethod;
specs.wo = gp01.wo;%specs.oEffect = oEff;

hyps.sf = sigf;hyps.sn = sign; hyps.ls = ls;
% save(saveFile,'error','t_pred','t_update','Ns','specs','runTime','hyps','Nll','oVar')

beep;