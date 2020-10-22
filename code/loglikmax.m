function hyp=loglikmax(kernel,hyp0,X,y,sn2)

ac=@(hyp)y*((kernel(X,X,hyp)+sn2*eye(size(X,2)))\y')+2*sum(log(diag(chol(kernel(X,X,hyp)+sn2*eye(size(X,2))))));

hyp=fmincon(ac,hyp0,[],[],[],[],ones(size(hyp0))*0.1,ones(size(hyp0))*100);

end