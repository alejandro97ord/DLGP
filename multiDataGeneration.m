clear all;close all;
select =3;
datasetSelect;

runTime = tic;
gp01 = mDLGPMopMean;
gp01.ard = 1;

gp01.meanFunction = {@(x)4,@(x)5,@(x)6,@(x)0,@(x)0,@(x)0,@(x) 0};
gp01.divMethod  = 3; %1: median, 2: mean, 3: mean(max, min)
gp01.wo = 2000; %overlapping factor

Y_train = Y_train(:,1:7);
Y_test = Y_test(:,1:7);
%data loaded from hyp.
% gp01.sigmaF = sigf; 
% gp01.sigmaN = sign;
% gp01.lengthS = ls;
gp01.outs = size(Y_train,2);

gp01.init(size(X_train,2),50,4000);
Nsteps = 1;
Ns = round([1,linspace(100,size(X_train,1),Nsteps)]);
%initialize GP
d = ['Initialized at : ',datestr(now,'HH.MM.SS')];
disp(d)
t_update = zeros(1,Nsteps);
t_pred = zeros(1,Nsteps);
output = zeros(size(Y_train,2),4449);
outvar = zeros(size(Y_train,2),4449);
negll = zeros(size(Y_train,2),4449);
rng(0);
for j = 0:Nsteps-1
    ave = Ns(j+2)-Ns(j+1); 
    tic;
    for p = Ns(j+1):Ns(j+2)-1
        gp01.update(X_train(p,:)',Y_train(p,:));
    end
    t_update(j+1) = toc/ave;
    
    tic;
    for d = 1: size(X_test,1)
%         output(:,d)=gp01.predict(X_test(d,:)');
%         [output(:,d),outvar(:,d)]=gp01.predictV(X_test(d,:)');
        [output(:,d),outvar(:,d),negll(:,d)]=gp01.predictL(X_test(d,:)',Y_test(d,:));
    end
%     oVar(DoF,j+1) = mean(outvar);
%     Nll(DoF,j+1) = mean(negll);
    t_pred(j+1) = toc/size(X_test,1);
%     error(DoF,j+1) = mean( (output'-Y_test(:,DoF)).^2 )/var(Y_test(:,DoF));
end
runTime = toc(runTime);
error = output' - Y_test;
disp(mean(error.^2)./var(Y_test));
disp(mean(outvar,2)');
disp(mean(negll,2)');
d = ['Finalized at: ',datestr(now,'HH.MM.SS')];
disp(d)
save('oB','error','t_pred','t_update','Ns','runTime','negll')
%%

res.K = gp01.K;
res.X = gp01.X;
res.Y = gp01.Y;
res.alpha = gp01.alpha;
res.invK = gp01.invK;
res.sigmaN = gp01.sigmaN;
res.lengthS = gp01.lengthS;
res.sigmaF = gp01.sigmaF;
res.dlik0 = gp01.dlik0;
res.delta = gp01.delta;

