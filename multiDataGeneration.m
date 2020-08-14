
clear all;close all;
select =3;
datasetSelect;
gp01 = mDLGPM;

gp01.divMethod  = 3; %1: median, 2: mean, 3: mean(max, min)
gp01.wo = 300; %overlapping factor

%data loaded from hyp.
gp01.sigmaF = sigf; 
gp01.sigmaN = sign;
gp01.lengthS = ls;
gp01.outs = size(Y_train,2);

gp01.init(size(X_train,2),50,10000);
Nsteps = 1;
Ns = round([1,linspace(100,size(X_train,1),Nsteps)]);
%initialize GP
d = ['Initialized at : ',datestr(now,'HH.MM.SS')];
disp(d)
t_update = zeros(1,100);
t_pred = zeros(1,100);
output = zeros(size(Y_train,2),4449);
for j = 0:Nsteps-1
    ave = Ns(j+2)-Ns(j+1); 
    tic;
    for p = Ns(j+1):Ns(j+2)-1
        gp01.update(X_train(p,:)',Y_train(p,:));
    end
    t_update(j+1) = toc/ave;
    
    tic;
    for d = 1: size(X_test,1)
        output(:,d)=gp01.predict(X_test(d,:)');
%         [output(d),outvar(d)]=gp01.predictL(X_test(d,:)');
%         [output(d),outvar(d),negll(d)]=gp01.predictL(X_test(d,:)',Y_test(d,DoF));
    end
%     oVar(DoF,j+1) = mean(outvar);
%     Nll(DoF,j+1) = mean(negll);
    t_pred(j+1) = toc/size(X_test,1);
%     error(DoF,j+1) = mean( (output'-Y_test(:,DoF)).^2 )/var(Y_test(:,DoF));
end
error = output' - Y_test;
disp(mean(error.^2)./var(Y_test));
d = ['Finalized at: ',datestr(now,'HH.MM.SS')];
disp(d)
