classdef RpropOnline < handle
    
    properties
        X;
        Y;
        
        lik;
        dlik0;
        delta;
        
        sigmaF;
        sigmaN;
        lengthS;
    end
    
    methods
        function obj = PR(obj)
            if isempty(obj.delta)
               obj.delta = 0.1*ones(size(obj.X,1)+2,1); 
            end

            if isempty(obj.sigmaF)
                x0 = ones(size(obj.X,1)+2,1);
                [~, obj.dlik0] = obj.grad(x0);
            else
                x0 = [obj.sigmaF;obj.sigmaN;obj.lengthS];
            end
            dmax = 50; dmin = 1e-6; etap = 1.2; etam = 0.5;
%             [~, obj.dlik0] = obj.grad(x0);
            
            x1  = x0 + sign (obj.dlik0).*obj.delta;
            if x1 (2) < 0.01
               x1(2) = 0.01; 
            end
            [fx1,dfx1] = obj.grad(x1);
            d1 = ((obj.dlik0.*dfx1)>0).*obj.delta*etap + ((obj.dlik0.*dfx1)<0).*obj.delta*etam + ...
                (obj.dlik0.*dfx1==0).*obj.delta;
            d1 = d1.*(d1>=dmin & d1 <= dmax)+(d1<dmin)*dmin + (d1>dmax)*dmax;
%             R(i) = fx0;

            obj.dlik0 = dfx1;
            obj.delta = d1;
            obj.sigmaF = x1(1);
            obj.sigmaN = x1(2);
            obj.lengthS = x1(3:end);
            obj.lik = fx1;
        end
        
        function [f, df] = grad(obj,hyps)
            df = zeros(size(hyps,1),1);
            dim = size(obj.X,2);
            obj.sigmaF = hyps(1);
            obj.sigmaN = hyps(2);
            obj.lengthS = hyps(3:end);
            ls = hyps(3:end);
            Ki = obj.fkernel(obj.X);
            Kn = Ki + eye(dim)*obj.sigmaN^2;
            alpha = Kn\obj.Y';
            auxDer = (alpha*alpha'-inv(Kn));
            df(1) = 0.5*trace( auxDer* (Ki.*2./obj.sigmaF) );
            df(2) = 0.5*trace( auxDer* eye(dim)*2*obj.sigmaN );
            
            for i = 1:size(ls)
                [k1,k2] = meshgrid(obj.X(i,:));
                df(2+i) =  0.5*trace( auxDer*( Ki.*((k1-k2).^2 ./ obj.lengthS(i)^3)) );
            end
            f = -0.5*obj.Y*(Kn\obj.Y') - 0.5*log(det(Kn)) - 0.5*size(Kn,1)*log(2*pi);
        end
        
        function kern = fkernel(obj, Xi)
            
            kern = zeros(size(Xi,2), size(Xi,2));
            for p = 1:size(Xi,1)
                [k1,k2] = meshgrid(Xi(p,:));
                kern = kern + (k1-k2).^2./obj.lengthS(p)^2;
            end
            kern = obj.sigmaF^2*exp(-0.5*kern);
        end
    end
end