function u = ctrlFeLi(t,x,p,ref)
%ctrlFeLi Feedback Linearization Controller with PD Controller
% IN: 
%   t     1 x 1
%   x     E x 1   position 1..E, velcoities E+1...2*E
%   p               Parameter struct
%    .f  @fun
%    .g  @fun
%    .K  E x E
%   ref  @fun 
% OUT: 
%   u  1 x 1
% Author: Jonas Umlauft, Last modified: 02/2017
x = x(:);
E =size(x,1);
if nargin >= 4
    r = ref(t); 
else
    r = zeros(E,1);
end
u = 1/p.g(x) * ( -p.f(x) - p.K*(x-r));


end