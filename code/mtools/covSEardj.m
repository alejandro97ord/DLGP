function [K,dKdhyp,dKdxi] = covSEardj(hyp,xi,xj,varargin)
%COVSEARDJ Squared Exponential covariance function with Automatic Relevance
% Detemination (ARD) distance measure. It is parameterized as:
%
% k(x,x') = sf^2 * exp(-0.5*sum((xi - xi')^2/li^2))
%
% hyp = [ log(l1); log(l2);..; log(sf) ]
%
% In:
%    xi            E  x n           Data points
%    xj            E  x m           Data points
%    hyp           E+1 x 1(1xE+1)   Hyperparameter
% Out:
%    K             n  x m        Covariance Matrix
%    dKdhyp        n  x m x E+1  Derivative wrt hyp
%    dKdxi         n x m x E x n  Derivative wrt xi
%
% E    : input dimensions
% n,m  : number of points
% Author: Jonas Umlauft, Last modified: 08/2015
% Test Code 
%{
clear, clc; rng default; E = 2; n = 3; m = 5; 
hyp = rand(E+1,1); xi = rand(E,n); xj = rand(E,m);
[r, num, ana] = checkGrad(@covSEardj,1,1,2,{hyp,xi,xj});
[r, num, ana] = checkGrad(@covSEardj,2,1,3,{hyp,xi,xj});
%}

hyp = hyp(:);
E = length(hyp)-1;
ell = exp(hyp(1:E));                               % characteristic length scale
sf2 = exp(2*hyp(E+1));

if isinf(sf2) || any(isinf(ell)), error('hyp is inf'); end

% Copy in case only one input is provided
if nargin < 3
    xj = xi;
end

% Verify Sizes
if size(xi,1)~=E, if size(xi,2)==E, xi=xi';
    else  error('size mismatch'); end
end

if size(xj,1)~=E, if size(xj,2)==E, xj=xj';
    else error('size mismatch'); end
end





il2= 1./(ell.^2);
ximxj = bsxfun(@minus,xi,permute(xj,[1 3 2]));
ximxj2il2 = permute(bsxfun(@times, ximxj.^2,il2),[2 3 1]);



K = sf2 * exp(-sum(ximxj2il2,3)./2);


% Derivatives wrt hyperparameter
if nargout > 1
    n = size(xi,2); m = size(xj,2);
    dKdhyp = zeros(n,m,E+1);
    % Derivative wrt ell
    for i=1:E
        dKdhyp(:,:,i) = K.*ximxj2il2(:,:,i);
    end
    % Derivative wrt sf
    dKdhyp(:,:,E+1) = 2*K;
end


% Derivatives wrt test inputs
if nargout > 2
    dKdxi =zeros(n,m,E,n);
    for i=1:n
        ximxjil2 = permute(bsxfun(@times, ximxj(:,i,:),il2),[1 3 2]);
        dKdxi(i,:,:,i) = permute(-bsxfun(@times, ximxjil2,K(i,:))',[3 1 2 4]);
    end
end


