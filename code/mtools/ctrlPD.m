function u = ctrlPD(t,x,p,ref)
%ctrlPD PD Controller for 2D
% IN: 
%   t     1 x 1
%   x     2*E x 1   position 1..E, velcoities E+1...2*E
%   p               Parameter struct
%    .Kp  E x E
%    .Kd  E x E      
%   ref  @fun 
% OUT: 
%   dxdt  E x 1
% Author: Jonas Umlauft, Last modified: 12/2016

[r, dr, ddr] = ref(t); 

E = size(x,1)/2; 
if mod(E,1)~=0
    error('Must have even number for size of state vector');
end
if size(r,1) ~= E || size(dr,1) ~= E || size(ddr,1) ~= E
    error('Reference has wrong dimension');
end
u =  - p.Kp*(x(1:E,:)-r) - p.Kd*(x(E+1:end,:)-dr) + ddr;

end
