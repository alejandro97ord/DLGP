
clear all;close all;%clc
% opengl('save','software')
select  = 3;
datasetSelect;
rng(0)
inputSize =size(X_train,2);
amountDoF = size(Y_train,2);

Nsteps = 100;
Ns = round([1,linspace(Nsteps,size(X_train,1),Nsteps)]);

output = zeros(amountDoF,size(X_test,1));
outvar = zeros(amountDoF,size(X_test,1));
negll = zeros(amountDoF,size(X_test,1));
Nll = zeros(amountDoF,Nsteps);
oVar= zeros(amountDoF,Nsteps);
t_update = zeros(1,Nsteps);
t_pred = zeros(1,Nsteps);
error = zeros(amountDoF,Nsteps);
start = strcat('Start time: ',datestr(now,'HH.MM.SS'));disp(start)
tic
runTime = tic;

% meanTorque;

d = ['Initialized test'];
disp(d)
rng(0)
gp01 = mDLGPMop;
gp01.ard = 1; %1 with ARD, 0 no ARD
gp01.divMethod  = 3; %1: median, 2: mean, 3: mean(max, min)
gp01.wo = 2000; %overlapping factor
gp01.outs = 7;
gp01.init(inputSize,50,4000);
for j = 0:Nsteps-1
    ave = Ns(j+2)-Ns(j+1);
    tic;
    for p =Ns(j+1):Ns(j+2)-1
        gp01.update(X_train(p,:)',Y_train(p,:)');
    end
    t_update(1,j+1) = toc/ave;
    tic;
    for d = 1: size(X_test,1)
        output(:, d)=gp01.predict(X_test(d,:)');
        %             [output(d),outvar(d)]=gp01.predictV(X_test(d,:)');
        %             [output(d),outvar(d),negll(d)]=gp01.predictL(X_test(d,:)',Y_test(d,DoF));
    end
    oVar(:,j+1) = mean(outvar,2 );
    Nll(:,j+1) = mean(negll, 2);
    t_pred(1,j+1) = toc/size(X_test,1);
    %     oEff(DoF,:) = gp01.oEffect; %used to register amount of data in Left, overlapping and right
    error(:,j+1) = mean( (output'-Y_test).^2 )./var(Y_test);
end

runTime = toc(runTime);

disp('    error      nll      var')
disp([error(:,end),Nll(:,end),oVar(:,end)]);
saveFile = strcat('testResults/',fileName,datestr(now,'mmm.dd_HH.MM.SS'),'oo.mat');

specs.pts = gp01.pts; specs.N = gp01.N; specs.divM = gp01.divMethod;
specs.wo = gp01.wo;%specs.oEffect = oEff;

save('mB','error','t_pred','t_update','Ns','specs','runTime','Nll','oVar')
save(saveFile)
% save(saveFile,'error','t_pred','t_update','Ns','specs','runTime','Nll','oVar')
%%
DoF = size(error,1);
figure(1)
for p = 1:DoF
    title('nMSe')
    subplot(2,4,p);semilogy(error(p,:),'LineWidth',2);hold on;
end
figure(2)
for p = 1:DoF
    title('NLL')
    subplot(2,4,p);plot(Nll(p,:),'LineWidth',2);hold on;
end
figure(3)
for p = 1:DoF
    title('Predict time')
    subplot(2,4,p);plot(t_pred(p,:),'LineWidth',2);hold on;
end
figure(4)
for p = 1:DoF
    title('Update time')
    subplot(2,4,p);plot(t_update(p,:),'LineWidth',2);hold on;
end
