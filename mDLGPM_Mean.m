classdef mDLGPM_Mean <handle
    %
    %{
    Data limit per GP: amount of data in a GP that triggers a division
    Limit of LGPs: max. amount of local GPs
    size k of vector x: [k,1] size of xTrain and xTests
    hyperplane method:
        1: mediand
        2: mean
        3: (max+min)/2
    if loadHyp:
        set fileName with variables:
            sigmaF
            sigmaN
            lengthScale (can be ARD or single value)
        inputs sigma_N, sigma_F, length-scale do not matter
    if !loadhyp
        set hyperparameters in sigma_N, sigma_F, length-scale
        fileName does not matter
    %}
    
    properties
        ard = 1;
        pts = 100; %Data limit per LGP
        N = 10; %maximum amount of LGPs
        xSize = 6;%x dimensionality
        kNum = 2; %amount of children a parent will divide into
        
        divMethod = 3; %Hyperplane method
        wo = 100; %Ratio Width/overlapping
        outs = 2;
        meanFunction = {@(x) x(1,:)+x(2,:)+x(3,:)+x(4,:)+x(5,:)+x(6,:),...
            @(x) x(1,:)+x(2,:)+x(3,:)+x(4,:)+x(5,:)+x(6,:)};
        sigmaF = 10 * ones(2,1) ; %sigma_F
        lengthS = 10 * ones(12,1); %lenght-scale
        sigmaN = 0.1 * ones(2,1); %sigma_N
    end
    
    properties%(Access = protected)
        %properties pts,X,Y,K,invK and alpha must be set to the desired
        %amount of training points
        count; %amount of local GPs
        localCount;%amount of data trained in each LGP
        X; %training samples
        Y; %training targets
        K; %covariance matrices
        alpha; %K\y
        L;
        auxAlpha;
        
        medians; %vector of hyperplanes
        parent; %vector of parent model
        children; %line 1: child 1 line 2: child 2 ...
        overlaps; %line 1:cutting dimension, line 2:end:size of overlapping region 1:end,
        auxUbic; %map a GP with the position of its data (K,L,alpha,X,Y,hyps,dlik0,delta)
        
        amountL; %if ard xSize else 1
    end
    
    methods
        function obj = init(obj, in_xSize,in_pts,in_N)
            %initialize data
            obj.pts = in_pts;
            obj.N = in_N;
            obj.xSize = in_xSize;
            obj.count = 1;
            
            obj.X = zeros(obj.xSize, obj.pts * obj.N);
            obj.Y = zeros(obj.outs, obj.pts * obj.N);
            obj.K = zeros(obj.pts*obj.outs, obj.pts * obj.N);
            obj.alpha = zeros(obj.pts*obj.outs,obj.N);
            obj.auxAlpha = zeros(obj.pts*obj.outs,obj.N);
            obj.L =zeros(obj.pts*obj.outs,obj.pts*obj.N);
            obj.localCount = zeros(1,2* obj.N -1);
            
            obj.medians =  zeros(obj.kNum - 1, 2*obj.N-1);
            
            obj.parent = zeros(1, 2 * obj.N-1);
            obj.children = -1*ones( obj.kNum , 2 * obj.N-1);
            
            obj.overlaps =  zeros( obj.kNum , 2 * obj.N-1);
            
            obj.auxUbic = zeros(1, 2 * obj.N-1);
            obj.auxUbic(1,1) = 1;
            
            if obj.ard == 1
                obj.amountL = obj.xSize;
                if size(obj.lengthS,1) ~= obj.xSize*obj.outs
                    error('Length scale must be a column of size input_dimensionality * number_of_outputs')
                end
            else
                obj.amountL = 1;
                if size(obj.lengthS,1) ~= obj.outs
                    error('Length scale must be a column of size number_of_outputs')
                end
            end
        end
        
        function kern = kernel(obj, Xi, Xj,out)%squared exponential kernel
            kern = (obj.sigmaF(out)^2)*...
                exp(-0.5*sum(((Xi-Xj).^2)./...
                (obj.lengthS((out-1)*obj.amountL+1:out*obj.amountL).^2),1))';
        end
        
        function m = mValue(obj, model,cutD)%compute the hyperplanes
            %evenly distributed
            m = 0 : obj.kNum;
            minX = min(obj.X(cutD, (obj.auxUbic(model)-1)*obj.pts+1:...
                obj.auxUbic(model)*obj.pts));
            dif = (max(obj.X(cutD, (obj.auxUbic(model)-1)*obj.pts+1:...
                obj.auxUbic(model)*obj.pts)) - minX) / obj.kNum ;
            m = minX + dif.*m;
            %             if obj.divMethod == 1
            %                 m = median(obj.X(cutD, (obj.auxUbic(model)-1)*obj.pts+1:...
            %                     obj.auxUbic(model)*obj.pts));
            %                 return
            %             elseif obj.divMethod == 2
            %                 m = mean(obj.X(cutD, (obj.auxUbic(model)-1)*obj.pts+1:...
            %                     obj.auxUbic(model)*obj.pts));
            %                 return
            %             elseif obj.divMethod == 3
            %                 m = (max(obj.X(cutD, (obj.auxUbic(model)-1)*obj.pts+1:...
            %                     obj.auxUbic(model)*obj.pts))+ min(obj.X(cutD, ...
            %                     (obj.auxUbic(model)-1)*obj.pts+1:...
            %                     obj.auxUbic(model)*obj.pts)))/2 ;
            %                 return
            %             end
        end
        
        function updateParam(obj,x,model)
            pos = obj.auxUbic(model)-1;
            if obj.localCount(model) == 1 %first point in model
                for p = 0:obj.outs-1
                    lH = chol(obj.kernel(x, x, p+1) + obj.sigmaN(p+1).^2);
                    obj.K(p*obj.pts+1,(pos)*obj.pts+1) = obj.kernel(x, x, p+1) + obj.sigmaN(p+1).^2;
                    obj.L(p*obj.pts+1,(pos)*obj.pts+1) = lH;
                    obj.alpha(p*obj.pts+1,pos+1) = lH'\(lH\(obj.Y(p+1,(pos)*obj.pts+1)- obj.meanFunction{p+1}(x)));
                    obj.auxAlpha(p*obj.pts+1,pos+1) = lH\(obj.Y(p+1,(pos)*obj.pts+1)- obj.meanFunction{p+1}(x));
                end
            else
                %set the updated parameters
                %auxX does not consider the new point x
                %auxY does consider the new point y
                auxX =  obj.X(:,(pos)*obj.pts+1:(pos)*obj.pts+obj.localCount(model)-1);
                auxY =  obj.Y(:,(pos)*obj.pts+1:(pos)*obj.pts+obj.localCount(model));
                for p = 0:obj.outs-1
                    b = obj.kernel(auxX,x,p+1);
                    c = obj.kernel(x,x,p+1)+obj.sigmaN(p+1)^2;
                    auxOut = p*obj.pts+1;
                    
                    newK = [obj.K(auxOut:auxOut+obj.localCount(model)-2,...
                        (pos)*obj.pts+1:(pos)*obj.pts+obj.localCount(model)-1),...
                        b;b',c];
                    
                    obj.K(auxOut:auxOut+obj.localCount(model)-1, (pos)*obj.pts+1:...
                        (pos)*obj.pts+obj.localCount(model)) = newK;
                    
                    obj.alpha(auxOut:auxOut+obj.localCount(model)-1,pos+1) =...
                        newK\( auxY( p+1 ,:) - obj.meanFunction{p+1}(auxX)' )';
                end
            end
        end
        
        function addPoint(obj, x, y, model)
            if obj.localCount(model) < obj.pts %if the model is not full
                obj.X(:,(obj.auxUbic(model)-1)*obj.pts+1+obj.localCount(model)) = x;
                obj.Y(:,(obj.auxUbic(model)-1)*obj.pts+1+obj.localCount(model)) = y;
                obj.localCount(model) = obj.localCount(model) + 1;
                obj.updateParam(x,model)
            end
            if obj.localCount(model) == obj.pts %if full
                div = 1;
                while div == 1 %divide until no child set has all the data
                    [div,model] = obj.divide(model);
                end
            end
        end
        
        function [div, childModel] =  divide(obj, model)
            if obj.parent(end)~= 0 %no memory for more divisions
                div = -1;
                childModel = -1;
                disp('no more divisions allowed')
            else
                %obtain cutting dimension
                [~,cutD]=max(max(obj.X(:, (obj.auxUbic(model)-1)*obj.pts+1:...
                    obj.auxUbic(model)*obj.pts),[],2)-min(obj.X(:, (obj.auxUbic(model)-1)*obj.pts+1:...
                    obj.auxUbic(model)*obj.pts),[],2));
                %obtain hyperplanes
                %first and last values of mP are max and min of X in cut
                %dimension
                mP = obj.mValue(model,cutD);
                %compute overlapping region
                if mP(end)-mP(1) == 0 %if all data is the same
                    o = 0.1; %overlapping size
                else
                    o  = ( mP(end)-mP(1) ) /obj.wo;
                end
                obj.medians(:,model)=mP(2:end-1)';
                obj.overlaps(1,model)=cutD;
                obj.overlaps(2,model)=o;
                
                distCount = zeros(1 , obj.kNum); %vector with amount of points in each new model
                distIndex = zeros(obj.pts , obj.kNum); %indeces of data in new models
                
                for i = 1:obj.pts
                    xD = obj.X(cutD,(obj.auxUbic(model)-1)*obj.pts+i);%x value in cut dimension
                    for j = 2 : obj.kNum
                        pL = 0;
                        if xD < mP(j) - o/2
                            pL = 1;
                        elseif xD >= mP(j) - o/2 && xD <= mP(j) + o/2
                            pL = 0.5 + (xD-mP(j))/(o);
                        end
                        if pL > rand() % if on left of current hyperplane
                            distCount( j-1 ) = distCount( j-1 )+1;
                            distIndex( distCount( j-1 ) ,j-1) = i;
                            break
                        elseif pL ~= 0 %if on right of current hyperplane
                            distCount( j ) = distCount( j )+1;
                            distIndex( distCount( j ) ,j) = i;
                            break
                        elseif pL == 0 && j == obj.kNum %when in last region
                            distCount( j ) = distCount( j )+1;
                            distIndex( distCount( j ) ,j) = i;
                            break
                        end
                    end
                end
                
                obj.localCount(model) = 0; %divided set is now "empty"
                if obj.count == 1
                    obj.count = obj.count+ + obj.kNum-1; %update the total number of sets
                else
                    obj.count = obj.count+ obj.kNum ;
                end
                obj.children(:,model) = obj.count - obj.kNum + 2 : obj.count+1; %assign the children
                obj.parent(obj.count - obj.kNum + 2 : obj.count+1) = model; %assign the parent
                
                %update parameters of new models
                obj.localCount(obj.count - obj.kNum + 2 : obj.count+1) = distCount;
                chAuxU = [obj.auxUbic(model),max(obj.auxUbic)+ (1:obj.kNum-1)];
                obj.auxUbic(obj.count - obj.kNum + 2 ) = obj.auxUbic(model);
                
                obj.auxUbic(obj.count - obj.kNum + 3 : obj.count+1)= max(obj.auxUbic)+ (1:obj.kNum-1);

                %update alpha and  L values for the new models
                B = (1:obj.pts);
                C = nonzeros(distIndex(1:end))';
                parentX = obj.X(:,(chAuxU(1)-1)*obj.pts+1: (chAuxU(1)-1)*obj.pts+obj.pts);
                parentY = obj.Y(:,(chAuxU(1)-1)*obj.pts+1: (chAuxU(1)-1)*obj.pts+obj.pts);
                for i = 0:obj.outs-1
                    newK = obj.K(i*obj.pts+1:(i+1)*obj.pts, (obj.auxUbic(model)-1)*obj.pts+1:...
                        (obj.auxUbic(model)-1)*obj.pts+obj.pts);
                    %permute K:
                    newK(B,:) =  newK(C,:);
                    newK(:,B) = newK(:,C);

                    %set child Ks
                    auxPos = 1; %determines where each child K is
                    for j = 1 :obj.kNum
                        
                        %sort child data
                        
                        obj.X(:,(chAuxU(j)-1)*obj.pts+1:(chAuxU(j)-1)*obj.pts+distCount(j)) = ...
                            parentX(:,distIndex(1:distCount(j), j));
                        obj.Y(:,(chAuxU(j)-1)*obj.pts+1:(chAuxU(j)-1)*obj.pts+distCount(j)) = ...
                            parentY(:,distIndex(1:distCount(j), j));
                        
                        
                        localK = newK(auxPos:auxPos + distCount(j)-1,auxPos:auxPos + distCount(j)-1);
                        obj.K(i*obj.pts+1:i*obj.pts+distCount(j)...
                            ,(chAuxU(j)-1)*obj.pts+1: (chAuxU(j)-1)*obj.pts+distCount(j)) = ...
                            localK;

                        %set child alphas
                        obj.alpha(i*obj.pts+1:i*obj.pts+distCount(j)...
                            , chAuxU(j)) =  localK \...
                            (parentY(i+1,distIndex(1:distCount(j), j))'-...
                            obj.meanFunction{i+1}(parentX(:,distIndex(1:distCount(j), j)))');
                        
                        auxPos = auxPos + distCount(j);
                    end
                end
                div = -1; %stop the divide loop
                childModel = -1;
                
                obj.auxUbic(model) = 0; %parent model will not have data
            end
        end
        
        function probs = activation(obj, x, model)
            probs = zeros(obj.kNum,1);
            if obj.children(1,model) == -1 %return zeros when model has no children
                return
            end
            mP = obj.medians(:, model); %hyperplane values
            xD = x(obj.overlaps(1,model)); %x value in cut dimension
            o = obj.overlaps(2,model); %overlapping region
            for j = 1 : obj.kNum-1
                pL = 0;
                if xD < mP(j) - o/2
                    pL = 1;
                elseif xD >= mP(j) - o/2 && xD <= mP(j) + o/2
                    pL = 0.5 + (xD-mP(j))/(o);
                end
                if pL ~= 0 % when in current hyperplane
                    probs(j) = pL;
                    probs(j+1) = 1-pL;
                    break
                elseif pL == 0 && j == obj.kNum-1 %when in last region
                    probs(j+1) = 1;
                    break
                end
            end
        end
        
        function update(obj,x,y)
            model = 1;
            while obj.children(1,model)~=-1 %while model is a parent
                %search for the leaf to asign the point
                probs = obj.activation(x, model);
                p0 = rand();
                for k = 1:length(probs)
                    if probs(k) >= p0
                        model = obj.children(k,model);%k-th child
                        break
                    end
                end
            end
            %add the model to the randomly selected model
            obj.addPoint(x,y,model)
        end
        
        function out = predict(obj,x)
            out = zeros(obj.outs,1);
            moP = zeros(2,1000);% line 1: active GPs, line 2: global probability
            mCount = 1; %number of GPs used for predictions
            moP(1,1) = 1; %start in root
            moP(2,1) = 1;
            %while all the GPs found are not leaves
            %get the GPs for prediction and thier global probabilites
            while ~isequal( obj.children(1,moP(1,1:mCount)) , -1*ones(1,mCount) )
                for j=1:mCount
                    probs = obj.activation(x,moP(1,j));
                    active = find(probs); %vector with active model indeces
                    if sum(probs==1) == 1 %if any child has prob of 1                        
                        moP(1,j) =  obj.children( active, moP(1,j));
                    elseif sum(probs) ~= 0 %if model has children
                        mCount = mCount + 1;
                        
                        moP(1,mCount) = obj.children(active(2),moP(1,j));
                        moP(2,mCount) = moP(2,j)*probs(active(2));
                        
                        moP(1,j) = obj.children(active(1),moP(1,j));
                        moP(2,j) = moP(2,j)*probs(active(1));
                    end
                end
            end
            %prediction: weigthing prediction with proabilities
            for p = 0:obj.outs-1
                for i=1:mCount
                    model = moP(1,i);
                    pred = obj.meanFunction{p+1}(x) + ...
                        ( obj.kernel(obj.X(:,(obj.auxUbic(model)-1)*obj.pts+1:...
                        (obj.auxUbic(model)-1)*obj.pts+obj.localCount(model)), x, p+1) )' * ...
                        obj.alpha(p*obj.pts+1: p*obj.pts+obj.localCount(model)...
                        ,obj.auxUbic(model));
                    out(p+1) = out(p+1)+pred*moP(2,i);
                end
            end
        end
        
        function [out, outVar] = predictV(obj,x)
            moP = zeros(2,1000);% line 1: active GPs, line 2: global probability
            mCount = 1; %number of GPs used for predictions
            moP(1,1) = 1; %start in root
            moP(2,1) = 1;
            %while all the GPs found are not leaves
            %get the GPs for prediction and thier global probabilites
            while ~isequal( obj.children(1,moP(1,1:mCount)) , -1*ones(1,mCount) )
                for j=1:mCount
                    [pL, pR] = obj.activation(x,moP(1,j));
                    if pL > 0 && pR == 0
                        moP(1,j) = obj.children(1,moP(1,j));
                        moP(2,j) = moP(2,j)*pL;
                    elseif pR > 0 && pL == 0
                        moP(1,j) = obj.children(2,moP(1,j));
                        moP(2,j) = moP(2,j)*pR;
                    elseif pL>0 && pR>0
                        mCount = mCount + 1;
                        moP(1,mCount) = obj.children(2,moP(1,j));
                        moP(2,mCount) = moP(2,j)*pR;
                        moP(1,j) = obj.children(1,moP(1,j));
                        moP(2,j) = moP(2,j)*pL;
                    end
                end
            end
            out = zeros(obj.outs,1);
            outVar = zeros(obj.outs,1);
            %prediction: weigthing prediction and variance with proabilities
            for p = 0:obj.outs-1
                for i=1:mCount
                    model = moP(1,i);
                    kxX = obj.kernel(obj.X(:,(obj.auxUbic(model)-1)*obj.pts+1:...
                        (obj.auxUbic(model)-1)*obj.pts+obj.localCount(model)), x, p+1);
                    pred = obj.meanFunction{p+1}(x) + ...
                        kxX' * obj.alpha(p*obj.pts+1: p*obj.pts+obj.localCount(model)...
                        ,obj.auxUbic(model));
                    out(p+1) = out(p+1)+pred*moP(2,i);
                    
                    v = obj.L(p*obj.pts+1:p*obj.pts+obj.localCount(model),...
                        (obj.auxUbic(model)-1)*obj.pts+1:(obj.auxUbic(model)-1)*...
                        obj.pts+obj.localCount(model))\kxX;
                    localVar = obj.kernel(x,x,p+1)- v'*v;
                    outVar(p+1)= outVar(p+1) + (localVar+pred^2)*moP(2,i);
                end
                outVar(p+1) = outVar(p+1) - out(p+1)^2;
            end
        end
        
        function [out, outVar,outLik] = predictL(obj,x,yTest)
            moP = zeros(2,900);% start from the root
            mCount = 1;
            moP(1,1) = 1;
            moP(2,1) = 1;
            while ~isequal( obj.children(1,moP(1,1:mCount)) , -1*ones(1,mCount) )
                for j=1:mCount
                    [pL, pR] = obj.activation(x,moP(1,j));
                    if pL > 0 && pR == 0
                        moP(1,j) = obj.children(1,moP(1,j));
                        moP(2,j) = moP(2,j)*pL;
                    elseif pR > 0 && pL == 0
                        moP(1,j) = obj.children(2,moP(1,j));
                        moP(2,j) = moP(2,j)*pR;
                    elseif pL>0 && pR>0
                        mCount = mCount + 1;
                        moP(1,mCount) = obj.children(2,moP(1,j));
                        moP(2,mCount) = moP(2,j)*pR;
                        moP(1,j) = obj.children(1,moP(1,j));
                        moP(2,j) = moP(2,j)*pL;
                    end
                end
            end
            out = zeros(obj.outs,1);
            outVar = zeros(obj.outs,1);
            outLik = zeros(obj.outs,1);
            for p =0:obj.outs-1
                for i=1:mCount
                    model = moP(1,i);
                    
                    kxX = obj.kernel(obj.X(:,(obj.auxUbic(model)-1)*obj.pts+1:...
                        (obj.auxUbic(model)-1)*obj.pts+obj.localCount(model)), x, p+1);
                    pred = obj.meanFunction{p+1}(x) + kxX' * obj.alpha(p*obj.pts+1: p*obj.pts+obj.localCount(model)...
                        ,obj.auxUbic(model));
                    out(p+1) = out(p+1)+pred*moP(2,i);
                    
                    v = obj.L(p*obj.pts+1:p*obj.pts+obj.localCount(model),...
                        (obj.auxUbic(model)-1)*obj.pts+1:(obj.auxUbic(model)-1)*...
                        obj.pts+obj.localCount(model))\kxX;
                    localVar = obj.kernel(x,x,p+1)- v'*v;
                    outVar(p+1)= outVar(p+1) + (localVar+pred^2)*moP(2,i);
                    
                    prob = normpdf(yTest(p+1),pred, sqrt(localVar + obj.sigmaN(p+1)^2));
                    outLik(p+1) = outLik(p+1)+max(prob,1e-300)*moP(2,i);
                end
                outVar(p+1) = outVar(p+1) - out(p+1)^2;
                outLik(p+1) = -log(outLik(p+1));
            end
        end
    end
end
