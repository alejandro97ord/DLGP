function dxdt = dynPendulum(t,x,ctrl,p)
%dynPendulum Dynamics of simple Pendulum
%   Continous time Pendulum model with 2D state x and 1D input u
% IN: 
%   t     1 x N
%   x     2 x N 
%   ctrl  @fun 
%       or 1 x N    numeric
%   p               Parameter struct
%    .a   1 x 1
%    .b   1 x 1
%    .c   1 x 1
% OUT: 
%   dxdt  2 x N
% Author: Jonas Umlauft, Last modified: 12/2016

[E,N]=size(x);
% Check dimensions
if E ~=2
    error('wrong input dimensions'); 
end

dxdt = zeros(2,N);
dxdt(1,:) = x(2,:);
if isnumeric(ctrl)
    if size(ctrl,2) ~=N, error('wrong input dimensions'); end
    dxdt(2,:) = -p.a*sin(x(1,:)) - p.b*x(2,:) + p.c*ctrl;
else
dxdt(2,:) = -p.a*sin(x(1,:)) - p.b*x(2,:) + p.c*ctrl(t,x);
end


end

