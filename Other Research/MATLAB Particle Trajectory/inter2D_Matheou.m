% number Fourier modes in each direction
Np = 2:2:64;
err_linf=zeros(size(Np));

for n = 1:length(Np)
    tic
    err_linf(n) = mkinterpolation2(Np(n),Np(n));
    t = toc;
    fprintf(1,'Np = %3d   Error = %12.6e   cpu = %g\n', Np(n), err_linf(n), t);
end
    
loglog(Np,err_linf,'o-')
xlabel('N');
ylabel('L_\infty error');


function [err_linf, v, v_err] = mkinterpolation2(M,N)

MN = N*M;

% domain size
Lx = 2*pi;
Ly = 2*pi;

% wavenumbers
kx=[0:M/2 -M/2+1:-1]*2*pi/Lx;
ky=[0:N/2 -N/2+1:-1]*2*pi/Ly;

% construct our test function
x = linspace(0,2*pi,M+1); x(end) = [];
y = linspace(0,2*pi,N+1); y(end) = [];

z = zeros(M,N);

for i = 1:M
    for j = 1:N
        z(i,j) = exp(sin(x(i))+sin(y(j)));
    end
end

% 2D DFT of our function. DFT is normalized
Z = fft2(z)/MN;

% grid for interpolation
im = 100;
jm = 100;

% interpolation locations (x,y)
xx = linspace(0,2*pi,im);
yy = linspace(0,2*pi,jm);

% interpolation result at (x,y)
v = zeros(im,jm);

% array with interpolation error
v_err = zeros(im,jm);

for ii = 1:im
    for jj = 1:jm
        v(ii,jj) = spectral_interpolation_2d(Z, kx, ky, xx(ii), yy(jj));
        
        v_err(ii,jj) = abs(v(ii,jj) - exp(sin(xx(ii))+sin(yy(jj))));
    end
end

err_linf = max(max(abs(v_err)));

end

function v = spectral_interpolation_2d(Z, kx, ky, x, y)

[M, N] = size(Z);

assert(M == length(kx))
assert(N == length(ky))

zx = zeros(1,M);

for i = 1:M
    for j=2:N
        zx(i) = zx(i) + Z(i,j)*exp(1i*ky(j)*y);
    end
    zx(i) = zx(i) + Z(i,1);
end

v = 0;
for i = 2:M
    v = v + zx(i)*exp(1i*kx(i)*x);
end
v = v + zx(1);

end


        