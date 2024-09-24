M = 256; % number of points
N = M;
Lx = 2*pi;
Ly = 2*pi;
nu = 5e-4;
Sc = 0.7;
beta = 0;
ar = 0.02; %random number amplitude
b = 1; %scalar gradient
CFLmax = 0.8;
tend = 200;
per = 1;

% PARTICLE TRACKING, ASSUMES 0 INITIAL V
np = 1; % number of tracked particles
xp = [3.5]; % start x positions
yp = [1]; % start y positions

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

phi = rand(size(u));
phihat = fft2(phi);


ncid = netcdf.create('turb2d16x8fs.nc','CLOBBER');

dimid_x = netcdf.defDim(ncid, 'x', M);
dimid_y = netcdf.defDim(ncid, 'y', N);
dimid_time = netcdf.defDim(ncid, 'time', netcdf.getConstant('NC_UNLIMITED'));

varid_x = netcdf.defVar(ncid, 'x', 'NC_FLOAT', [dimid_x]);
varid_y = netcdf.defVar(ncid, 'y', 'NC_FLOAT', [dimid_y]);
varid_u = netcdf.defVar(ncid, 'u', 'NC_FLOAT', [dimid_x, dimid_y, dimid_time]);
varid_v = netcdf.defVar(ncid, 'v', 'NC_FLOAT', [dimid_x, dimid_y, dimid_time]);
varid_omega = netcdf.defVar(ncid, 'vorticity', 'NC_FLOAT', [dimid_x, dimid_y, dimid_time]);
varid_phi = netcdf.defVar(ncid, 'scalar', 'NC_FLOAT', [dimid_x, dimid_y, dimid_time]);
varid_dissipation = netcdf.defVar(ncid, 'dissipation', 'NC_FLOAT', [dimid_x, dimid_y, dimid_time]);
varid_time = netcdf.defVar(ncid, 'time', 'NC_FLOAT', [dimid_time]);
netcdf.endDef(ncid);

netcdf.putVar(ncid, varid_time, 0, 1, time);
netcdf.putVar(ncid, varid_x, 0, M, x);
netcdf.putVar(ncid, varid_y, 0, N, y);
netcdf.putVar(ncid, varid_phi, [0, 0, 0], [M, N, 1], phi);
netcdf.putVar(ncid, varid_u, [0, 0, 0], [M, N, 1], u);
netcdf.putVar(ncid, varid_v, [0, 0, 0], [M, N, 1], v);
netcdf.putVar(ncid, varid_omega, [0, 0, 0], [M, N, 1], omega);
netcdf.putVar(ncid, varid_dissipation, [0, 0, 0], [M, N, 1], 0*omega);

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
    factp = exp(-nu/Sc*8/15*dt*k2);
    
    r0o = -fft2(u.*omegadx+v.*omegady)+beta*vhat;
    r0p = -fft2(u.*real(ifft2(ddx.*phihat))+v.*real(ifft2(ddy.*phihat)))+b*vhat;
    
    omegahat = facto.*(omegahat + dt*8/15*r0o); % update omega
    phihat = factp.*(phihat + dt*8/15*r0p); % update phi
    
    
    %%%% Substep 2
    psihat = -idel2.*omegahat;
    uhat = ddy.*psihat;
    vhat = -ddx.*psihat;
    
    u = real(ifft2(uhat));
    v = real(ifft2(vhat));
    
    omegadx = real(ifft2(ddx.*omegahat));
    omegady = real(ifft2(ddy.*omegahat));
    
    r1o = -fft2(u.*omegadx+v.*omegady)+beta*vhat;
    r1p = -fft2(u.*real(ifft2(ddx.*phihat))+v.*real(ifft2(ddy.*phihat)))+b*vhat;
    
    omegahat = omegahat + dt*(-17/60*facto.*r0o + 5/12*r1o);
    phihat = phihat + dt*(-17/60*factp.*r0p + 5/12*r1p);
    facto = exp(-nu*(-17/60+5/12)*dt*kk);
    factp = exp(-nu/Sc*(-17/60+5/12)*dt*k2);
    omegahat = omegahat.*facto;
    phihat = phihat.*factp;
    
    %%%% Substep 3
    psihat = -idel2.*omegahat;
    uhat = ddy.*psihat;
    vhat = -ddx.*psihat;
    
    % max(max(abs(real(ifft2(1i*ddx.*uhat+1i*ddy.*vhat))))) % divergence
    
    u = real(ifft2(uhat));
    v = real(ifft2(vhat));
    
    omegadx = real(ifft2(ddx.*omegahat));
    omegady = real(ifft2(ddy.*omegahat));
    
    % PARTICLE TRACKING SECTION START
    % up = inter_2D(x, y, u', [xp, yp]); % particle u velocity
    % vp = inter_2D(x, y, v', [xp, yp]); % particle v velocity

    % xp = mod(xp + up * dt, 2 * pi);
    % yp = mod(yp + vp * dt, 2 * pi);
    
    % PARTICLE TRACKING SECTION END

    r2o = -fft2(u.*omegadx+v.*omegady)+beta*vhat;
    r2p = -fft2(u.*real(ifft2(ddx.*phihat))+v.*real(ifft2(ddy.*phihat)))+b*vhat;    
    omegahat = omegahat + dt*(-5/12*facto.*r1o + 3/4*r2o);
    phihat = phihat + dt*(-5/12*factp.*r1p + 3/4*r2p);
    facto = exp(-nu*(-5/12+3/4)*dt*kk);
    factp = exp(-nu/Sc*(-5/12+3/4)*dt*kk);
    omegahat = omegahat.*facto;
    phihat = phihat.*factp;
    
    
    phihat = filter.*phihat;
    omegahat = filter.*omegahat;
    
    CFL = max(max(abs(u)))/dx*dt+max(max(abs(v)))/dy*dt;
    dt = CFLmax/CFL*dt; %0.0005;
        
    time = time + dt;
    nstep = nstep + 1;
    
    if mod(nstep,20)==0
        phi = real(ifft2(phihat));
        omega = real(ifft2(omegahat));
        dissipation = 2*nu*(real(ifft2(ddx.*uhat)).^2 + real(ifft2(ddy.*uhat)).^2 + real(ifft2(ddx.*vhat)).^2 + real(ifft2(ddy.*vhat)).^2);
        eta = (nu^3/mean(dissipation,'all'))^0.25;
        
        uv_mag = (u .^ 2 + v .^ 2) .^ 0.5;
        contourf(x, y, uv_mag); shading flat; axis equal; colorbar; hold on;
        % scatter(xp, yp, 50, "red", "filled");
        everyOther = 8;
        quiver(x(1:everyOther:end), y(1:everyOther:end), u(1:everyOther:end, ...
            1:everyOther:end), v(1:everyOther:end, 1:everyOther:end)); axis equal; hold off; drawnow

        % subplot(221); pcolor(x,y,omega'); shading flat; axis equal; colorbar; drawnow
        % title("omega")
        % subplot(222); pcolor(x,y,phi'); shading flat; axis equal; colorbar; drawnow
        % title("phi")
        % subplot(223); pcolor(x,y,dissipation'); shading flat; axis equal; colorbar; drawnow
        % title("dissipation")
        % subplot(223); pcolor(x,y,u); shading flat; axis equal; colorbar; drawnow
        % subplot(224); pcolor(x,y,v); shading flat; axis equal; colorbar; drawnow
        % title("u")
        
        fprintf(1,'step = %d    time = %g    dt = %g  CFL = %g    kmax*eta = %g %g\n', nstep, time, dt, CFL, eta.*kmax, eta.*kmax/sqrt(Sc));
        
        [~, dim_time_len] = netcdf.inqDim(ncid,dimid_time);
        
        netcdf.putVar(ncid, varid_time, [dim_time_len], [1], time);
        netcdf.putVar(ncid, varid_phi, [0, 0, dim_time_len], [M, N, 1], phi);
        netcdf.putVar(ncid, varid_u, [0, 0, dim_time_len], [M, N, 1], u);
        netcdf.putVar(ncid, varid_v, [0, 0, dim_time_len], [M, N, 1], v);
        netcdf.putVar(ncid, varid_omega, [0, 0, dim_time_len], [M, N, 1], omega);
        netcdf.putVar(ncid, varid_dissipation, [0, 0, dim_time_len], [M, N, 1], dissipation);
        netcdf.sync(ncid);
    end
end

netcdf.close(ncid);
