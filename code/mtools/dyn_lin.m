function dxdt = dyn_lin(~,x,A)
% Dynamics for a linear system
% In: 
%    t    1 x 1     time (not used)
%    x    n x 1     state
%    A    n x n     state transition matrix
% Out: 
%    dxdt n x 1     state derivative
dxdt = A*x;

