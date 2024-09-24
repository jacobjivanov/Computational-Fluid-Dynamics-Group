%f=@(x) sin(2*x);
%f=@(x) (x > pi) - sin(x) + (x < sin(5.5 * x) - log(x));
%f=@(x) log(x)-2* sin(x);
%f=@(x) 1./(1+(x-pi).^2);
f=@(x) log(x+5);

L=2*pi;

np=50; % even number of points ONLY
x=linspace(0,2*pi,np);
% x(end)=[];

y = f(x);
Y=fft(y)/np;

kx(1:np/2+1)=(0:np/2)*2*pi/L;
kx(np:-1:np/2+2)=(-1:-1:-np/2+1)*2*pi/L;

xx=linspace(0,L,1001);
yy=zeros(size(xx));

% yy interpolates (x,y) points
yy(:)=real(Y(1));
for i=2:np
    v=real(Y(i))*cos(kx(i)*xx)-imag(Y(i))*sin(kx(i)*xx);
    yy=yy+v;
end

plot(x,y,'o',xx,yy,'r',xx,f(xx),'b','LineWidth',3);