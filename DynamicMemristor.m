function [I,G]=DynamicMemristor(V,G,para)
G=para.r*G+(1-para.r)*para.G0+updata(V,para.alpha).*(binaryFunc(V)-G);
I=G.*(para.Kp*NL(max(V,0))+para.Kn*NL(min(V,0)));
end

function y=binaryFunc(x)
id=x>0;
y(id,1)=1;
y(~id,1)=0;
end

function y=updata(x,a)
y=a*abs(x)./(a*abs(x)+1);
end

function y=NL(x)
y=x.^3;
end
