
clear all;close all;clc
% opengl('save','software')
fileName = "KUKA_flask";


%dataset and hyperparameters in the same folder.
if ~ exist('X_train','var') || ~ exist('Y_train','var')
    load('C:\Users\alejandro\Desktop\P10\rt-learning-with-dlgp\benchmarks\data\KUKA_flask.mat')
end

inputSize =size(X_train,2);
amountDoF = size(Y_train,2);

Nsteps = 100;
Ns = round([1,linspace(100,size(X_train,1),Nsteps)]);

output = zeros(1,size(X_train,1));
nllMeans = zeros(amountDoF,size(X_train,1));
varMeans= zeros(amountDoF,size(X_train,1));
t_update = zeros(amountDoF,size(X_train,1));
t_pred = zeros(amountDoF,size(X_train,1));
error = zeros(amountDoF,size(X_train,1));
rng(0)
runTime = tic;

for DoF = 3:3%1:5
    d = ['Initialized DoF ',num2str(DoF),' at: ',datestr(now,'HH.MM.SS')];
    disp(d)
    clear gp01
    gp01 = mDLGPop(inputSize,50,2*50000);
    gp01.ard = 1; %1 with ARD, 0 no ARD
    gp01.divMethod  = 3; %1: median, 2: mean, 3: mean(max, min)
    gp01.wo = 300; %overlapping factor
    for j = 1:size(X_train,1)

        tic;
        [output(j),varMeans(DoF,j),nllMeans(DoF,j)]=gp01.predictL(X_train(j,:)',Y_train(j,DoF));
        t_pred(DoF,j) = toc;
        
        error(DoF,j) = (output(j)-Y_train(j,DoF)).^2 /var(Y_train(:,DoF));
        
        tic;
        gp01.update(X_train(j,:)',Y_train(j,DoF));
        t_update(DoF,j+1) = toc;
    end
end

runTime = toc(runTime);
error2 =  cumsum(error,2)./linspace(1,size(nllMeans,2),size(nllMeans,2));
nll2 =  cumsum(nllMeans,2)./linspace(1,size(nllMeans,2),size(nllMeans,2));

disp('    o-error      o-nll    ')
disp([error2(:,end),nll2(:,end)]);
saveFile = strcat('testResults/',fileName,datestr(now,'mmm.dd_HH.MM.SS'),'oo.mat');

specs.pts = gp01.pts; specs.N = gp01.N; specs.divM = gp01.divMethod;
specs.wo = gp01.wo;%specs.oEffect = oEff;
% cumsum(error,2)./linspace(1,size(nllMeans,2),size(nllMeans,2));
save(saveFile,'error','t_pred','t_update','specs','runTime','nllMeans','varMeans')
%%
close all
DoF = size(error,1);
window = 10000;
figure(1)
for p = 1:DoF
    title('nMSe')
    subplot(2,4,p);semilogy(movmean(error(p,:),window),'LineWidth',2);hold on;
end
figure(2)
for p = 1:DoF
    title('NLL')
    subplot(2,4,p);plot(movmean(nllMeans(p,:),window),'LineWidth',2);hold on;
end
figure(3)
for p = 1:DoF
    title('Predict time')
    subplot(2,4,p);plot(movmean(t_pred(p,:),window),'LineWidth',2);hold on;
end
figure(4)
for p = 1:DoF
    title('Update time')
    subplot(2,4,p);plot(movmean(t_update(p,:),window),'LineWidth',2);hold on;
end
