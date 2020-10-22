function K=kernel(x1,x2,hyp)

K=zeros(size(x1,2),size(x2,2));
for i=1:size(x1,2)
    K(i,:)=hyp(1)^2*exp(-sum((x1(:,i)-x2).^2,1)./(2*hyp(2)^2));
end

end