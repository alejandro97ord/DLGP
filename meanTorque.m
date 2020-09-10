constant = 1.9;
Iz1 = constant;   %moment of inertia
Iz2 =constant;    
m1 = constant;    %mass
m2 = constant;
l1 = constant;    %length
l2 = constant;
r1 = l1;   %length to the center of mass
r2 = l2;
g = 9.81;  %gravity

a = Iz1 + Iz2 + m1*r1^2 + m2*( l1^2 + r2^2 );
b = m2*l1*r2;
d = Iz2 + m2*r2^2;
      

tau = @(x) (a + 2*b*cos( x(2,:) )).*x(5,:) +  (d + b*cos( x(2,:) )).*x(6,:)...
     +(-b*sin( x(2,:) ).*x(4,:)).*x(3,:) + (-b*sin( x(2,:) ).*( x(3,:) + x(4,:) )).*x(4,:)...
       +m1*g*r1*cos( x(1,:) ) + m2*g*( l1*cos( x(1,:) ) + r2*cos( x(1,:)+x(2,:) ) );
   
tau2 =@(x) (d + b*cos( x(2,:) )).*x(5,:) +              d   *x(6,:)...
     +b*sin( x(2,:) ).*x(3,:).*x(3,:)...
     + m2*g*r2*cos( x(1,:)+x(2,:) );
func  = {tau,tau2};
