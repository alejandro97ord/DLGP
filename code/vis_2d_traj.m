addpath(genpath('../code/mtools'));addpath(genpath('../code/matlab2tikz'));
plotname = [path_plot, setname];
plot2tex=true;



%% Snapshot at maximum variance
figure;hold on;axis equal; xlim([-r r]);ylim([-r r]); 
% [~,imaxs2] = max(sigfun(Xsim))
imaxs2 = 60
z = linspace(0,1,100);
Xhullmax = Xte(:,ihull{imaxs2})/r*(r-2)+Xd(:,imaxs2);
fill(Xhullmax(1,:),Xhullmax(2,:),'r');
% plot(Xtr(1,:),Xtr(2,:),'*g');
Xmaxs2 = Xsim(:,1:imaxs2); Xdmaxs2= Xd(:,1:imaxs2);
plot(Xmaxs2(1,:),Xmaxs2(2,:),'b');
plot(Xdmaxs2(1,:),Xdmaxs2(2,:),'g-');

%% Snapshot at minimal variance
figure;hold on;axis equal; xlim([XteMin(1) XteMax(2)]);ylim([XteMin(2) XteMax(2)])
% [~,imins2] = min(sigfun(Xsim(:,imaxs2:end))); imins2 = imaxs2+imins2
imins2 = 209
z = linspace(0,1,100);
Xhullmin = Xte(:,ihull{imins2})/r*(r-2)+Xd(:,imins2);
fill(Xhullmin(1,:),Xhullmin(2,:),'y');
% plot(Xtr(1,:),Xtr(2,:),'*g');
Xmins2 = Xsim(:,1:imins2); Xdmins2 = Xd(:,1:imins2);
plot(Xmins2(1,:),Xmins2(2,:),'b');
plot(Xdmins2(1,:),Xdmins2(2,:),'g-');



%% Animation
% figure;  hold on; xlim([XteMin(1) XteMax(2)]);ylim([XteMin(2) XteMax(2)]);
% for it=1:length(T)
%     if it>1, delete(htext); delete(h); end
%     Xhull = Xte(:,ihull{it});
%     h = fill(Xhull(1,:),Xhull(2,:),'r');
%     plot(Xd(1,it),Xd(2,it),'k*');
%     plot(X(1,it),X(2,it),'ob');   
%     htext =  text(Xd(1,it),Xd(2,it)-1,num2str(AreaError(it)));
%     plot(Xtr(1,:),Xtr(2,:),'*g');
%     drawnow;
%     pause(0.02);
% end

% % %% Less test data
% Nte = 1e3; Ndte = floor(nthroot(Nte,E));  Nte = Ndte^E;
% Xtes = ndgridj([-4;-4], [4;4],Ndte*ones(E,1)) ;
% Xtes1 = reshape(Xtes(1,:),Ndte,Ndte); Xtes2 = reshape(Xtes(2,:),Ndte,Ndte);


% Plot tracking error
norme = sqrt(sum((Xsim-Xd).^2,1));
figure;  xlabel('t'); ylabel('|e|'); title('tracking error');
semilogy(T,norme);
hold on;
emax = sqrt(e2max);
semilogy(T,emax);


%% Plot dphi1
plotname = [path_plot, setname];
plot2tex=true;

figure; hold on; axis tight; xlabel('x1'); ylabel('x2');axis equal;
title(' rhog');

rhog(1)=rhof(1);%phi1(1)-min(phi2'-rhof);

surf(Xtes1,Xtes2,reshape(phi1'-rhog,Ndte,Ndte),'EdgeColor','none','FaceColor','interp'); colormap(jet);
% surf(Xtes1,Xtes2,reshape(phibar1te/max(phibar1te)-phi1'/max(phi1),Ndte,Ndte),'EdgeColor','none','FaceColor','interp'); colormap(jet);
if plot2tex
    matlab2tikz_table([plotname, '_rhog_surf.tex'],'showInfo',false);
end


% Plot dphi2
figure; hold on; axis tight; xlabel('x1'); ylabel('x2');axis equal;
title(' rhof');
surf(Xtes1,Xtes2,reshape(phi2'-rhof,Ndte,Ndte),'EdgeColor','none','FaceColor','interp'); colormap(jet);
% surf(Xtes1,Xtes2,reshape(phibar2te/max(phibar2te)-phi2'/max(phi2),Ndte,Ndte),'EdgeColor','none','FaceColor','interp'); colormap(jet);
if plot2tex
    matlab2tikz_table([plotname, '_rhof_surf.tex'],'showInfo',false);
end

%% Save to txt file
if plot2tex
    opts.var_names = {'T','Xsim','Xtr','Xd','Xhullmax','Xhullmin','Xmaxs2','Xmins2','Xdmaxs2','Xdmins2','norme','emax'};
    vars2txt       = { T , Xsim', Xtr', Xd', Xhullmax', Xhullmin', Xmaxs2', Xmins2', Xdmaxs2', Xdmins2',norme', emax'};
    opts.fname = plotname;
    opts.ndata = 2000;
    data2txt(opts,vars2txt{:});
    clear vars2txt
end
