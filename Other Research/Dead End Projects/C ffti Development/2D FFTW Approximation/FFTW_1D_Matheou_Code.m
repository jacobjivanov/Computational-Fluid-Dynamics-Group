% f=@(x) sin(2*x);
f=@(x) x > pi;
% f=@(x) -x.*(x-2*pi);
% f=@(x) 1./(1+(x-pi).^2);

L = 2 * pi;

np = 18; % even number of points ONLY
x = linspace(0, 2 * pi, np + 1);
x(end)=[];

y = f(x);
Y = fft(y) / np;

kx(1:np / 2 +1) = (0:np / 2) * 2 * pi / L;
kx(np:-1:np/2+2)=(-1:-1:-np/ 2 + 1) * 2 * pi / L;

xx = linspace(0, L, 1001);
yy = zeros(size(xx));

% yy interpolates (x,y) points
yy(:)=real(Y(1));
for i = 2:np
    v = real(Y(i)) * cos(kx(i) * xx) - imag(Y(i)) * sin(kx(i) * xx);
    yy = yy + v;
end

yy(122 + 1)
yy(369 + 1)
yy(385 + 1)
yy(508 + 1)
yy(640 + 1)
yy(699 + 1)
yy(710 + 1)
yy(811 + 1)
yy(884 + 1)
yy(996 + 1)

plot(x,y,'o',xx,yy,'r',xx,f(xx),'b','LineWidth',3);