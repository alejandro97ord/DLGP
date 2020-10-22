function [norm_diff, dfdx_num, dfdx_ana] = checkGrad(fcn,N_in,N_out,N_grad,inargs,e)
% This function compares the analytic and numeric Jacobian (dydx)
% of a function from input x of dimension P1 x P1 x ... to output 
% y of dimension Q1 x Q1 x ... using finite differenzes 
% Thus dydx is P1 x P1 x ... x Q1 x Q1 x ...
% 
% In: 
%   fcn    handle    handle to the function to be checkes
%   N_in   1 x 1     index of input argument x  in the function
%   N_out  1 x 1     index of output y in the function
%   N_grad 1 x 1     index of gradient  dydx in the function
%   iargs  cell array of dimension nargin of the function
%   (e      1 x 1     intervall used for finite differenzes)
% Out: 
%   norm_diff 1 x 1  norm of difference between numeric and analytic should  be around 1e-7
%   dfdx_num  P1 x P1 x ... x Q1 x Q1 x ...  Numeric Jacobian
%   dfdx_ana  P1 x P1 x ... x Q1 x Q1 x ...  Analyitc Jacobian
% Last modified: 2016/03/12, Jonas Umlauft

if nargin < 6
    e = 1e-8;
end

x = inargs{N_in};
out = cell(1,max(N_out,N_grad));

[out{:}] = feval(fcn,inargs{:});

dfdx_ana = out{N_grad};
y = out{N_out};
sizex = size(x);
x = x(:);
dfdx_num =zeros(numel(y),numel(x));
for i=1:numel(x)
    dx = x;
    dx(i) = x(i) + e;
    inargs{N_in} = reshape(dx,sizex);
    
    [out{:}] = feval(fcn,inargs{:});
    dy = out{N_out};
    dfdx_num(:,i) = (dy(:)-y(:))./e;
end

dfdx_num = reshape(dfdx_num,size(dfdx_ana));
norm_diff = norm(dfdx_ana(:)-dfdx_num(:))/norm(dfdx_num(:));
format long
disp('       Numeric                 Analytic           Differenz')
disp([dfdx_num(:) dfdx_ana(:) dfdx_num(:)-dfdx_ana(:) ])
disp(norm_diff);
format short


