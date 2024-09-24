x = linspace(0, 2 * pi, 100);
y = linspace(0, 2 * pi, 100);

[X, Y] = meshgrid(x, y);
u = sin(X);
v = cos(Y);

xp = 4; % particle start x position
yp = 1; % particle start y position

time = 0;
nstep = 0;
dt = 0.01;
tend = 100;
while time < tend
    up = inter_2D(x, y, u', [xp, yp]); % particle u velocity
    vp = inter_2D(x, y, v', [xp, yp]); % particle v velocity
    
    xp = mod(xp + up * dt, 2 * pi);
    yp = mod(yp + vp * dt, 2 * pi);
    
    if mod(nstep, 10) == 0
        scatter(xp, yp, 50, "red", "filled"); hold on
        everyOther = 8;
        quiver(x(1:everyOther:end), y(1:everyOther:end), ...
            u(1:everyOther:end, 1:everyOther:end), v(1:everyOther:end, 1:everyOther:end));
        xlim([0, 2 * pi])
        ylim([0, 2 * pi])
        axis equal;
        hold off
        drawnow
    end
    nstep = nstep + 1;
    time = time + dt;
end