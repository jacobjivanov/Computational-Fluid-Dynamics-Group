M = 64; % number of points
N = M;
Lx = 2*pi;
Ly = 2*pi;
nu = 1e-3;
Sc = 0.7;
beta = 0;
ar = 0.05; %random number amplitude
b = 1; %scalar gradient
CFLmax = 0.8;
tend = 100;


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

[u, v, omega, psi, ddx, ddy, idel2, kk, k2]=deal(zeros(M,N));
[vx, vy, ax, ay, uprev, vprev]=deal(zeros(M,N));


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
        u(i,j) =  cos(x(i))*sin(y(j))+ar*rand;
        v(i,j) = -sin(x(i))*cos(y(j))+ar*rand;
    end
end

uhat = fft2(u);
vhat = fft2(v);
omegahat = ddx.*vhat - ddy.*uhat; % make vorticity 


phi = rand(size(u));
phihat = fft2(phi);

psihat = -idel2.*omegahat;

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


% animation = VideoWriter('particles2','MPEG-4'); %open video file
% animation.FrameRate = 30; 
% open(animation)


dt = min([dx dy])/20.;

nstep = 1;



while time < tend

    omegadx = real(ifft2(ddx.*omegahat));
    omegady = real(ifft2(ddy.*omegahat));
        
    r0o = -fft2(u.*omegadx+v.*omegady)-nu*k2.*omegahat;
    r0p = -fft2(u.*real(ifft2(ddx.*phihat))+v.*real(ifft2(ddy.*phihat)))-nu/Sc*k2.*phihat;
    
    omegahat = omegahat + dt*8/15*r0o; % update omega
    phihat =   phihat + dt*8/15*r0p; % update phi
    
    psihat = -idel2.*omegahat;
    uhat = ddy.*psihat;
    vhat = -ddx.*psihat;
    
    u = real(ifft2(uhat));
    v = real(ifft2(vhat));
    
    
    %%%% Substep 2 %%%%
    
    omegadx = real(ifft2(ddx.*omegahat));
    omegady = real(ifft2(ddy.*omegahat));
    
    r1o = -fft2(u.*omegadx+v.*omegady)-nu*k2.*omegahat;
    r1p = -fft2(u.*real(ifft2(ddx.*phihat))+v.*real(ifft2(ddy.*phihat)))-nu/Sc*k2.*phihat;
    
    omegahat = omegahat + dt*(-17/60*r0o + 5/12*r1o);
    phihat = phihat + dt*(-17/60*r0p + 5/12*r1p);
    
    psihat = -idel2.*omegahat;
    uhat = ddy.*psihat;
    vhat = -ddx.*psihat;
    
    % max(max(abs(real(ifft2(1i*ddx.*uhat+1i*ddy.*vhat))))) % divergence
    
    u = real(ifft2(uhat));
    v = real(ifft2(vhat));
    
    
    %%%% Substep 3 %%%%
    
    omegadx = real(ifft2(ddx.*omegahat));
    omegady = real(ifft2(ddy.*omegahat));

    r2o = -fft2(u.*omegadx+v.*omegady)-nu*k2.*omegahat;
    r2p = -fft2(u.*real(ifft2(ddx.*phihat))+v.*real(ifft2(ddy.*phihat)))-nu/Sc*k2.*phihat;    
    omegahat = omegahat + dt*(-5/12*r1o + 3/4*r2o);
    phihat = phihat + dt*(-5/12*r1p + 3/4*r2p);
    
    psihat = -idel2.*omegahat;
    uhat = ddy.*psihat;
    vhat = -ddx.*psihat;
    
    u = real(ifft2(uhat));
    v = real(ifft2(vhat));

    
    
    % Filter 
    phihat = filter.*phihat;
    omegahat = filter.*omegahat;

    % advance time
    time = time + dt;
    nstep = nstep + 1;

    
    % CFL and next time step
    CFL = max(max(abs(u)))/dx*dt+max(max(abs(v)))/dy*dt;
    %dt = 0.005; %CFLmax/CFL*dt; %0.0005;
        
    
    if mod(nstep,20)==0
        phi = real(ifft2(phihat));
        omega = real(ifft2(omegahat));
        dissipation = 2*nu*(real(ifft2(ddx.*uhat)).^2 + real(ifft2(ddy.*uhat)).^2 + real(ifft2(ddx.*vhat)).^2 + real(ifft2(ddy.*vhat)).^2);
        eta = (nu^3/mean(dissipation,'all'))^0.25;
        
        %uv_mag = (u .^ 2 + v .^ 2) .^ 0.5;
        
        pcolor(x, y, omega'); shading flat; axis equal; axis tight; xlabel('x'); ylabel('y'); drawnow;
        
        %caxis([-10 10])
        %colorbar off 
        %colormap redblue
        %set(gcf,'color','w');
        

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
        
%         [~, dim_time_len] = netcdf.inqDim(ncid,dimid_time);
%         
%         netcdf.putVar(ncid, varid_time, [dim_time_len], [1], time);
%         netcdf.putVar(ncid, varid_phi, [0, 0, dim_time_len], [M, N, 1], phi);
%         netcdf.putVar(ncid, varid_u, [0, 0, dim_time_len], [M, N, 1], u);
%         netcdf.putVar(ncid, varid_v, [0, 0, dim_time_len], [M, N, 1], v);
%         netcdf.putVar(ncid, varid_omega, [0, 0, dim_time_len], [M, N, 1], omega);
%         netcdf.putVar(ncid, varid_dissipation, [0, 0, dim_time_len], [M, N, 1], dissipation);
%         netcdf.sync(ncid);
    end
end

%close(animation);

netcdf.close(ncid);
