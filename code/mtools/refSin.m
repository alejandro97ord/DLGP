function [r, dr, ddr] = refSin(t)
%refSin Sinosoidal reference function
%   returns up to 2nd derviative
% Author: Jonas Umlauft, Last modified: 12/2016


r = sin(t) ; 
dr = cos(t); 
ddr = -sin(t);

end

