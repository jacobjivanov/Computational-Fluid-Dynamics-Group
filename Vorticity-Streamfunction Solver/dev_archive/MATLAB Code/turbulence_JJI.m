M = 256; % number of points
N = M;
Lx = 2*pi;
Ly = 2*pi;
nu = 5e-4;
Sc = 0.7;
beta = 0;
ar = 0; % 0.02; % random number amplitude
b = 1; % scalar gradient
CFLmax = 0.8;
tend = 200;
per = 1;

x=linspace(0,Lx,M+1); x(end)=[];
dx = Lx/M;
kx=[0:M/2 -M/2+1:-1]*2*pi/Lx;

y=linspace(0,Ly,N+1); y(end)=[];
ky=[0:N/2 -N/2+1:-1]*2*pi/Ly;
dy = Ly/N;

time = 0;

index_kmax = ceil(M/3);
kmax = kx(index_kmax);
filter = ones(M,N);
filter(index_kmax+1:2*index_kmax+3,index_kmax+1:2*index_kmax+3)=0;

rng(64);

[u, v, omega, psi, ddx, ddy, idel2, kk, k2] = deal(zeros(M,N));

for j=1:N
    ddx(:,j)=1i*kx;
end

for i=1:M
    ddy(i,:)=1i*ky;
end

for i=1:M
    for j=1:N
        idel2(i,j)=-kx(i)^2-ky(j)^2;
    end
end
idel2=1./idel2;
idel2(1,1)=0;

for i=1:M
    for j=1:N
        kk(i,j)=kx(i)^2+ky(j)^2;
        k2(i,j)=kx(i)^2+ky(j)^2;
        if kk(i,j) >= 6^2 && kk(i,j) <= 7^2 % forcing 
            kk(i,j) = -kk(i,j); 
        end
         
        if kk(i,j) <= 2^2
            kk(i,j) = 8*kk(i,j); % large-scale dissipation
        end
     end
end


for i=1:M
    for j=1:N
        u(i,j) =  cos(per*x(i))*sin(per*y(j))+ar*rand;
        v(i,j) = -sin(per*x(i))*cos(per*y(j))+ar*rand;
    end
end

uhat = fft2(u);
vhat = fft2(v);
omegahat = ddx.*vhat - ddy.*uhat; % make vorticity 

dt = 0.5*min([dx dy]);

nstep = 1;

while time < tend
    psihat = -idel2.*omegahat;
    uhat = ddy.*psihat;
    vhat = -ddx.*psihat;
    
    u = real(ifft2(uhat));
    v = real(ifft2(vhat));
    
    omegadx = real(ifft2(ddx.*omegahat));
    omegady = real(ifft2(ddy.*omegahat));
    
    facto = exp(-nu*8/15*dt*kk);
    r0o = -fft2(u.*omegadx+v.*omegady)+beta*vhat;
    omegahat = facto.*(omegahat + dt*8/15*r0o); % update omega
    
    %%%% Substep 2
    psihat = -idel2.*omegahat;
    uhat = ddy.*psihat;
    vhat = -ddx.*psihat;
    
    u = real(ifft2(uhat));
    v = real(ifft2(vhat));
    
    omegadx = real(ifft2(ddx.*omegahat));
    omegady = real(ifft2(ddy.*omegahat));
    
    r1o = -fft2(u.*omegadx+v.*omegady)+beta*vhat;
    
    omegahat = omegahat + dt*(-17/60*facto.*r0o + 5/12*r1o);
    facto = exp(-nu*(-17/60+5/12)*dt*kk);
    omegahat = omegahat.*facto;
    
    %%%% Substep 3
    psihat = -idel2.*omegahat;
    uhat = ddy.*psihat;
    vhat = -ddx.*psihat;
        
    u = real(ifft2(uhat));
    v = real(ifft2(vhat));
    
    omegadx = real(ifft2(ddx.*omegahat));
    omegady = real(ifft2(ddy.*omegahat));

    r2o = -fft2(u.*omegadx+v.*omegady)+beta*vhat;
    omegahat = omegahat + dt*(-5/12*facto.*r1o + 3/4*r2o);
    facto = exp(-nu*(-5/12+3/4)*dt*kk);
    omegahat = omegahat.*facto;    
    
    omegahat = filter.*omegahat;
    
    CFL = max(max(abs(u)))/dx*dt+max(max(abs(v)))/dy*dt;
    dt = CFLmax/CFL*dt; %0.0005;
        
    time = time + dt;
    nstep = nstep + 1;
    
    if mod(nstep,1)==0
        omega = real(ifft2(omegahat));
        dissipation = 2*nu*(real(ifft2(ddx.*uhat)).^2 + real(ifft2(ddy.*uhat)).^2 + real(ifft2(ddx.*vhat)).^2 + real(ifft2(ddy.*vhat)).^2);
        eta = (nu^3/mean(dissipation,'all'))^0.25;
        omega_TG = zeros(M, N);
        for i = 1:M
            for j = 1:N
                omega_TG(i, j) = -2*per*exp(-2*nu*time)*cos(per*x(i))*cos(per*y(j));
            end
        end
        contourf(x, y, omega - omega_TG); colorbar; axis equal; drawnow;

        % uv_mag = (u .^ 2 + v .^ 2) .^ 0.5;
        % contourf(x, y, uv_mag); shading flat; axis equal; colorbar; hold on;
        % everyOther = 8;
        % quiver(x(1:everyOther:end), y(1:everyOther:end), u(1:everyOther:end, ...
        %    1:everyOther:end), v(1:everyOther:end, 1:everyOther:end)); axis equal; hold off; drawnow
        
        fprintf(1,'step = %d    time = %g    dt = %g  CFL = %g    kmax*eta = %g %g\n', nstep, time, dt, CFL, eta.*kmax, eta.*kmax/sqrt(Sc));
                
    end
end