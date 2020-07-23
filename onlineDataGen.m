
clear all;close all;%clc
% opengl('save','software')
select = 3;
if select == 1
    fileName = "Real_Sarcos";
    load('C:\Users\alejandro\Desktop\P10\rt-learning-with-dlgp\benchmarks\data\Real_Sarcos.mat')
elseif select == 2
    fileName = "Real_Barrett";
elseif select == 3
    fileName = "Real_Sarcos_long";
    load('C:\Users\alejandro\Desktop\P10\rt-learning-with-dlgp\benchmarks\data\Real_Sarcos_long.mat')
elseif select == 4
    fileName = "SL_Sarcos";
elseif select == 5
    fileName = "SL_Barrett";
elseif select == 6
    fileName = "2DoFData_large";
    load('C:\Users\alejandro\Desktop\P10\rt-learning-with-dlgp\benchmarks\data\2DoFData_large.mat')
elseif select == 7
    fileName = "KUKA_flask";
end

%dataset and hyperparameters in the same folder.
if ~ exist('X_train','var') || ~ exist('Y_train','var')
    load(fileName)
end
rng(0)
inputSize =size(X_train,2);
amountDoF = size(Y_train,2);

Nsteps = 100;
Ns = round([1,linspace(100,size(X_train,1),Nsteps)]);

output = zeros(1,size(X_test,1));
outvar = zeros(1,size(X_test,1));
negll = zeros(1,size(X_test,1));
Nll = zeros(amountDoF,100);
oVar= zeros(amountDoF,100);
t_update = zeros(amountDoF,100);
t_pred = zeros(amountDoF,100);
error = zeros(amountDoF,100);
start = strcat('Start time: ',datestr(now,'HH.MM.SS'));disp(start)
tic
runTime = tic;
% meanTorque;
for DoF = 1:amountDoF
    d = ['Initialized DoF: ',num2str(DoF)];
    disp(d)
    gp01 = mDLGPopMean;
    gp01.ard = 1; %1 with ARD, 0 no ARD
    gp01.divMethod  = 3; %1: median, 2: mean, 3: mean(max, min)
    gp01.wo = 300; %overlapping factor
    gp01.init(inputSize,50,2*50000);
    for j = 0:Nsteps-1
        ave = Ns(j+2)-Ns(j+1);
        tic;
        for p =Ns(j+1):Ns(j+2)-1
            gp01.update(X_train(p,:)',Y_train(p,DoF));
        end
        t_update(DoF,j+1) = toc/ave;
        tic;
        for d = 1: size(X_test,1)
%             output(d)=gp01.predict(X_test(d,:)');
%             [output(d),outvar(d)]=gp01.predictV(X_test(d,:)');
            [output(d),outvar(d),negll(d)]=gp01.predictL(X_test(d,:)',Y_test(d,DoF));
        end
        oVar(DoF,j+1) = mean(outvar);
        Nll(DoF,j+1) = mean(negll);
        t_pred(DoF,j+1) = toc/size(X_test,1);
        %     oEff(DoF,:) = gp01.oEffect; %used to register amount of data in Left, overlapping and right
        error(DoF,j+1) = mean( (output'-Y_test(:,DoF)).^2 )/var(Y_test(:,DoF));
    end
end
runTime = toc(runTime);

disp('    error      nll      var')
disp([error(:,end),Nll(:,end),oVar(:,end)]);
saveFile = strcat('testResults/',fileName,datestr(now,'mmm.dd_HH.MM.SS'),'oo.mat');

specs.pts = gp01.pts; specs.N = gp01.N; specs.divM = gp01.divMethod;
specs.wo = gp01.wo;%specs.oEffect = oEff;

save(saveFile,'error','t_pred','t_update','Ns','specs','runTime','Nll','oVar')
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
