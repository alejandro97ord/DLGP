clearvars; clear; close all; clc;rng default; addpath(genpath('./mtools'));
path_plot= ''; path_data= '';
plot2tex = true;  setname = '2D_traj';

pp=parcluster('local');
if(isempty(gcp('nocreate')))
    parpool(pp.NumWorkers,'IdleTimeout',240);
end

%% Set Parameters
disp('Setting Parameters...')

% Basic Parameters
steps = 5;
runs = steps*4;
Ntr = 1000;    % Number of training points
Tsim = 30;   % Simulation time
Nsim = 1000;  % Simulation steps
sn = 0.5 ;     % Observation noise (std deviation)
E = 2;        % State space dimension

% Initial State /reference for simulation
x0 = [0 0]';
% ref = @(t) refGeneral(t,E+1,@(tau) sigmf(tau,[10 50]));  % step
% ref = @(t) refGeneral(t,E+1,@(tau) zeros(size(tau)));  % zeros
ref = @(t) refGeneral(t,E+1,@(tau) 2*sin(tau));  % circle


% Controller gains
pFeLi.lam = ones(E-1,1);
% pFeLi.kc = 2;

% Define Systemdynamics
a = 1; b = 1; c = 0; d=20;
pdyn.f = @(x) 1-sin(x(1,:)) + b*sigmf(x(2,:),[a c]);
pdyn.g = @(x) d*(1+0.5*sin(0.25*x(2,:)));

fhat = @(x)0;
ghat = @(x)d;


% GP learning and simulation  parameters
optGPR = {'KernelFunction','squaredexponential','ConstantSigma',true,'Sigma',sn};
odeopt = odeset('RelTol',1e-3,'AbsTol',1e-6);
% ls = [4 4];  sf = 2;
% optGPR = {'KernelFunction','ardsquaredexponential','fitMethod','none',...
%     'OptimizeHyperparameters','none','KernelParameters',[ls sf], 'ConstantSigma',true,'Sigma',sn};

% Visualization
r = 2.5;
Nte = 1e4; XteMin = [-r -r]; XteMax = [r r];
Ndte = floor(nthroot(Nte,E));  Nte = Ndte^E;
Xte = ndgridj(XteMin, XteMax,Ndte*ones(E,1)) ;
Xte1 = reshape(Xte(1,:),Ndte,Ndte); Xte2 = reshape(Xte(2,:),Ndte,Ndte);
Ntrajplot = 100;

% Lyapunov test
tau=0.0001;     % Grid distance
delta = 0.01;     % Probability for error bound
deltaL = 0.01;     % Probability for Lipschitz constant

%%  Initialize Hyperparameters
disp('Initialize Hyperparameters...')
pFeLi.f = fhat; pFeLi.g = ghat;
pFeLi.kc=1000;
dyn = @(t,x) dynAffine(t,x,@(t,x) ctrlFeLi(t,x,pFeLi,ref),pdyn);
xt0 = ref(0);
xt0 = xt0(1:2);
[T,Xtr] = ode45(dyn,linspace(0,2*pi,Ntr),xt0); Xtr= Xtr';
Utr = zeros(1,length(T));
for i=1:length(T)
    Utr(i) = ctrlFeLi(T(i),Xtr(:,i),pFeLi,ref);
end
Ytr = pdyn.f(Xtr) - fhat(Xtr) + (pdyn.g(Xtr) - ghat(Xtr)).*Utr +  sn.*randn(1,Ntr);
hyps = loglikmax(@compKern,ones(4,1),[Xtr;Utr],Ytr,sn^2);
% gprModel = fitrgp(Xtr',Ytr',optGPR{:});
% ls = exp(gprModel.Impl.ThetaHat(1));  sf = exp(gprModel.Impl.ThetaHat(end));


% %%  Generate Uniformly Distributed Training Points
% disp('Generating Training Points...')
% Ntr = ceil(10^(idx/steps));
% pFeLi.f = fhat; pFeLi.g = ghat;
% pFeLi.kc=1000;
% dyn = @(t,x) dynAffine(t,x,@(t,x) ctrlFeLi(t,x,pFeLi,ref),pdyn);
% xt0 = ref(0);
% xt0 = xt0(1:2);
% [Ttr,Xtr] = ode45(dyn,linspace(0,2*pi,Ntr),xt0); Xtr= Xtr';
% Utr = zeros(1,size(Xtr,2));
% for i=1:size(Xtr,2)
%     Utr(i) = ctrlFeLi(Ttr(i),Xtr(:,i),pFeLi,ref);
% end
% Ytr = pdyn.f(Xtr) - fhat(Xtr) + (pdyn.g(Xtr) - ghat(Xtr)).*Utr +  sn.*randn(1,Ntr);


%% Learn Model - Optimize Hyperparameters
Ksf =  compKern([Xtr;Utr],[Xtr;Utr],hyps)+sn^2*eye(Ntr);
alpha = Ksf\Ytr';
muf = @(x) fhat(x)+kernel(x,Xtr,hyps(1:2))*alpha;
sig2f = @(x) diag(hyps(1)^2-kernel(x,Xtr,hyps(1:2))*(Ksf\kernel(x,Xtr,hyps(1:2))'));
mug = @(x) ghat(x)+(kernel(x,Xtr,hyps(3:4)).*Utr)*alpha;
sig2g = @(x) diag(hyps(3)^2-(kernel(x,Xtr,hyps(3:4)).*Utr)*(Ksf\(kernel(x,Xtr,hyps(3:4)).*Utr)'));
pFeLi.kc = 40;
pFeLi.f = @(x)fhat(x)+muf(x);
pFeLi.g = @(x)ghat(x)+mug(x);



%% Compute Lipschitz Constants
disp('Computing Lipschitz Constants...')

Lf =  max(sqrt(sum(gradestj(pdyn.f,Xte).^2,1)));
Lg =  max(sqrt(sum(gradestj(pdyn.g,Xte).^2,1)));


kf = @(x,xp) hyps(1)^2 * exp(-0.5*sum((x-xp).^2./hyps(2).^2,1));
kg = @(x,xp) hyps(3)^2 * exp(-0.5*sum((x-xp).^2./hyps(4).^2,1));
dkfdxi = @(x,xp,i)  -(x(i,:)-xp(i,:))./hyps(2)^2 .* kf(x,xp);
dkgdxi = @(x,xp,i)  -(x(i,:)-xp(i,:))./hyps(4)^2 .* kg(x,xp);

dmufdxi = @(x,i) dkfdxi(x,Xtr,i)*alpha;
dmugdxi = @(x,i) dkgdxi(x,Xtr,i)*(Utr'.*alpha);
Kcov = inv(Ksf);
dsig2fdxi = @(x,i) -2*kf(x,Xtr)*Kcov*dkfdxi(x,Xtr,i)';
dsig2gdxi = @(x,i) -2*(((kg(x,Xtr).*Utr)*Kcov).*Utr)*dkfdxi(x,Xtr,i)';

Lmuf = zeros(Nte,1);
Lsig2f = zeros(Nte,1);
dmufdx = zeros(Nte,E);
dsig2fdx = zeros(Nte,E);
Lmug = zeros(Nte,1);
Lsig2g = zeros(Nte,1);
dmugdx = zeros(Nte,E);
dsig2gdx = zeros(Nte,E);
parfor i = 1:Nte
    for e=1:E
        dmufdx(i,e) = dmufdxi(Xte(:,i),e);
        dsig2fdx(i,e) = dsig2fdxi(Xte(:,i),e);
        dmugdx(i,e) = dmugdxi(Xte(:,i),e);
        dsig2gdx(i,e) = dsig2gdxi(Xte(:,i),e);
    end
end

Lmuf = 2*max(sqrt(sum(dmufdx.^2,2)));
Lsig2f = 2*max(sqrt(sum(dsig2fdx.^2,2)));
Lmug = 2*max(sqrt(sum(dmugdx.^2,2)));
Lsig2g = 2*max(sqrt(sum(dsig2gdx.^2,2)));


%% Test Lyapunov condition
disp('Setup Lyapunov Stability Test...')
tau=0.0001;

eta = pFeLi.kc/(pFeLi.kc+pFeLi.lam);
omegaf = sqrt(Lsig2f*tau);
omegag = sqrt(Lsig2g*tau);
beta = 2*log((1+((max(XteMax)-min(XteMin))/tau))^E/delta);
gammaf = tau*(Lmuf+2*Lf) + sqrt(beta)*omegaf;
gammag = tau*(Lmug+2*Lg) + sqrt(beta)*omegag;

alphabar = @(X)(sqrt(beta.*sig2g(X))+gammag)./mug(X);

Lyapincr = @(X,t) sqrt(sum((X-nonzeros((ref(t)+1e-10).*[1;1;0])).^2,1))' <= (sqrt(beta.*sig2f(X))+gammaf+...
    abs(nonzeros((ref(t)+1e-10).*[0;0;1])-muf(X)).*(sqrt(beta.*sig2g(X))+gammag)./mug(X))./...
    (pFeLi.kc*(1-(sqrt(beta.*sig2g(X))+gammag)./(eta*mug(X))));



%% Simulate System with Feedback Linearization and PD Controller
disp('Simulation...')
dyn = @(t,x) dynAffine(t,x,@(t,x) ctrlFeLi(t,x,pFeLi,ref),pdyn);
[T,Xsim] = ode45(dyn,linspace(0,Tsim,Nsim),x0); Xsim= Xsim';
Xd = ref(T);Xd = Xd(1:E,:);
AreaError = zeros(Nsim,1); ihull = cell(Nsim,1);

all(alphabar(Xte)'<eta|sqrt(sum(Xte.^2,1))>r)
parfor nsim=1:Nsim
    ii = find(Lyapincr(Xte/r*(r-2)+Xd(:,nsim),T(nsim)));
    try
        ihull{nsim} = ii(convhull(Xte(:,ii)','simplify',true));
        AreaError(nsim) = polyarea(Xte(1,ihull{nsim}),Xte(2,ihull{nsim}));
    catch
        AreaError(nsim) = 0;
    end
    e2max(nsim) = max(sum((Xte(:,ii)/r*(r-2)).^2,1));
end

ebsig = max(sqrt(e2max));
maxV=max(AreaError);

%% Data Quality Assessment
disp('Data Assessment...')
Xddt = ref(T);

% reduce number of test samples
Nte = 1e4; Ndte = floor(nthroot(Nte,E));  Nte = Ndte^E;
Xtes = ndgridj([-r;-r], [r;r],Ndte*ones(E,1)) ;
Xtes1 = reshape(Xtes(1,:),Ndte,Ndte); Xtes2 = reshape(Xtes(2,:),Ndte,Ndte);


M=1;
Uabs = sort(abs(Utr),'ascend');
uup = Uabs(round(0.1*length(Uabs)));
ulow = Uabs(round(0.9*length(Uabs)));

for i=1:Nte
    d2 = sum((Xtes(:,i)-Xtr).^2,1);
    [d2,idx] = sort(d2,'ascend');
    U = Utr(idx);
    d2_1 = d2(abs(U)>ulow);
    d2_2 = d2(abs(U)<uup);
    phi1(i) = d2_1(M);
    phi2(i) = d2_2(M);
end

dVe = @(e)sqrt(diag(e'*diag([pFeLi.lam, 1])*e));
xi_f = @(e) 0.25*pFeLi.kc*diag(e'*diag([pFeLi.lam, 1])*e);
cx = @(x,idx) abs(Xddt(end,idx)-muf(x));
pi_ctrl = @(x,idx) (pFeLi.kc/eta*sqrt(diag((x-Xddt(1:end-1,idx))'*diag([pFeLi.lam, 1])*(x-Xddt(1:end-1,idx))))+cx(x,idx))./mug(x);


phibarf = @(x,idx)-hyps(2)^2*log(1-(xi_f(x-Xddt(1:end-1,idx))-gammaf*dVe(x-Xddt(1:end-1,idx))).^2./(beta*hyps(1)^2*dVe(x-Xddt(1:end-1,idx)).^2));
phibarg = @(x,idx)-hyps(4)^2*log(1-(xi_f(x-Xddt(1:end-1,idx))-gammag*dVe(x-Xddt(1:end-1,idx)).*pi_ctrl(x,idx)).^2./...
    (beta*hyps(3)^2*dVe(x-Xddt(1:end-1,idx)).^2.*pi_ctrl(x,idx).^2));

rhog = 1000*ones(size(Xtes,2),1);
rhof = 1000*ones(size(Xtes,2),1);
for nsim=1:Nsim
    pbg = phibarg(Xtes,nsim);
    rhog = min(rhog,pbg); 
    
    pbf = phibarf(Xtes,nsim);
    pbf(imag(pbf)~=0) = max(phi2);
    rhof = min(rhof,pbf);
end

% phibar1 = @(x)-hyps(4)^2*log((1-(min(alphabar(Xte(:,sqrt(sum(Xte.^2,1))<r)))*mug(x)-gammag).^2/(beta*hyps(3)^2)));
% phibar1te = phibar1(Xtes);
% phibar2 = @(x,idx) -hyps(2)^2*log(1-(0.5*pFeLi.kc-gammaf-...
%     min(alphabar(Xtes(:,sqrt(sum(Xtes.^2,1))<r)))*abs(Xddt(end,idx)-muf(x))).^2/(beta*hyps(1)^2));
% phibar2te = 100*ones(size(Xtes,2),1);
% for nsim=1:Nsim
%     pb2 = phibar2(Xtes,nsim);
%     phibar2te = min(pb2,phibar2te); 
% end


%% Viualization and Saving
disp('Plotting Results and Saving...')
monitor_memory_whos
% vis_2d_traj;
distFig();
% save([path_data setname],'maxV','maxVeb','steps','runs','eb','ebsig');
save('test');
disp('Pau');