function y = HenonMap(x) 
wn=normrnd(0,0.05^2,1,x);
y=zeros(1,x+2);
for i=3:x+2
    y(i)=1-1.4*y(i-1)^2+0.3*y(i-2)+wn(i-2);
end
y=y(3:end);
end