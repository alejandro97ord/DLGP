function U1 = cholupdatej(U,a,b)
%CHOLUPDATEJ: Computes upper triangular cholesky for U1 = chol [U'*U a: a'  b])
% In: 
%   U : N x N   upper triangular
%   a : N x 1
%   b   1 x 1
% Out: 
%   U1  N+1 x N+1
%{
clc; clear; rng default;
N=5e3; K1 = randn(N); K1 = K1'*K1; K = K1(1:end-1,1:end-1); a = K1(1:end-1,end); b = K1(end,end); 
U = chol(K); 
tic, U1 = cholupdatej(U,a,b); toc
tic, U1t = chol(K1);  toc
norm(U1-U1t)
%}
a=a(:);
n = size(a,1);

if size(U,1) ~= n || size(U,2)~=n || ~isscalar(b)
    error('wrong input dimensions'); 
end

Ua = U'\a;

U1 = [U Ua; zeros(1,n) sqrt(b-a'*(U\(Ua)))];

                