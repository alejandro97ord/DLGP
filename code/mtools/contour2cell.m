function contcell = contour2cell(cont)
% Converts the contour matrix obtained by [C,~] = countour() to a cell
% where each cell is one level line
%In: 
%   cont  2 X N
% Out: 
%   contcell {M} 2 x ? 
pos=1;i=1;pot=[];
while pos<size(cont,2)
    pot=[pot;cont(:,pos(i)+1:cont(2,pos(i))+pos(i))'];
    pos=[pos;cont(2,pos(i))+pos(i)+1];
    i=i+1;
end
contcell=cell(length(pos)-1,1);
for i=1:length(pos)-1
    contcell{i}=pot(pos(i)-i+1:pos(i+1)-i-1,:);
end
