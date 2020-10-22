function  [Lvec, dLvecdSPD] = SPD2Lvec(SPD)
% Performs Cholesky decomposition SPD = L*L'  for N symmetric positive 
% matrices and writes in vector
% In:
%    SPD      D x D x N  symmetric positive definit matrix
% Out:
%    Lvec       triD x N
%    dLvecdSPD triD x N x D x D x N
%{
clear, clc, rng default; addpath('./mtools');
D = 2; N=3; A = rand(D); SPD = bsxfun(@times,A*A',rand(1,1,N));
[r, num, ana] = checkGrad(@SPD2Lvec,1,1,2,{SPD});
%}
[D,E,N] = size(SPD); if E~=D, error('SPD not square');end
triD = (D+1)*D/2;
Lvec= zeros(triD,N); dLvecdSPD = zeros(triD, N,D,D,N);
iiL = tril(true(D)); 
[L, dLdSPD]  = cholj(SPD);

for n=1:N
    Ltemp =L(:,:,n);
   Lvec(:,n) = Ltemp(iiL);
   for d1 = 1:D 
       for d2 = 1:D; 
              dLdSPDtemp = dLdSPD(:,:,n,d1,d2,n);
              dLvecdSPD(:,n,d1,d2,n) = dLdSPDtemp(iiL);
       end
   end
end
end
