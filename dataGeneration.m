%gp01 = mDLGP(xSize,pts,N);
tic
gp01 = mDLGP;

gp01.divMethod  = 3; %1: median, 2: mean, 3: mean(max, min)
gp01.wo = 2000; %overlapping factor

%data loaded from hyp.
gp01.sigmaF = sigf(DoF); 
gp01.sigmaN = sign(DoF);
gp01.lengthS = ls(:,DoF);

gp01.init(inputSize,50,10000);

%initialize GP
disp('initialized')

for j = 0:Nsteps-1
    ave = Ns(j+2)-Ns(j+1); 
    tic;
    for p = Ns(j+1):Ns(j+2)-1
        gp01.update(X_train(p,:)',Y_train(p,DoF));
    end
    t_update(DoF,j+1) = toc/ave;
    tic;
    for d = 1: size(X_test,1)
%         output(d)=gp01.predict(X_test(d,:)');
%         [output(d),outvar(d)]=gp01.predictV(X_test(d,:)');
        [output(d),outvar(d),negll(d)]=gp01.predictL(X_test(d,:)',Y_test(d,DoF));
    end
    oVar(DoF,j+1) = mean(outvar);
    Nll(DoF,j+1) = mean(negll);
    t_pred(DoF,j+1) = toc/size(X_test,1);
%     oEff(DoF,:) = gp01.oEffect; %used to register amount of data in Left, overlapping and right
    error(DoF,j+1) = mean( (output'-Y_test(:,DoF)).^2 )/var(Y_test(:,DoF));
end

