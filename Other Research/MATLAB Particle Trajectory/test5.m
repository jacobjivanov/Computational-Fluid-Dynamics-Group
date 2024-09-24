N = 8;
N_inter = 64;
x_max = 100 * rand();
y_max = 100 * rand();
x = linspace(0, x_max, N);
y = linspace(0, y_max, N);

[X,Y] = ndgrid(x,y);

normX = X .* 2 .* pi ./ x_max;
normY = Y .* 2 .* pi ./ y_max;

values2D = exp(sin(normX) + sin(normY));

x_inter = linspace(0, x_max, N_inter);
y_inter = linspace(0, y_max, N_inter);

[X_inter,Y_inter] = ndgrid(x_inter,y_inter);

normX_inter = X_inter .* 2 .* pi ./ x_max;
normY_inter = Y_inter .* 2 .* pi ./ y_max;

values2D_inter = interp2(x, y, values2D,X_inter,Y_inter,"linear");
values2D_exact = exp(sin(normX_inter) + sin(normY_inter));
values2D_error = values2D_inter - values2D_exact;

pcolor(X_inter, Y_inter, values2D_error); 
colorbar;
clim([-1,1]);
colormap(redblue)
shading interp

% https://www.mathworks.com/matlabcentral/fileexchange/25536-red-blue-colormap
function c = redblue(m)
    %REDBLUE    Shades of red and blue color map
    %   REDBLUE(M), is an M-by-3 matrix that defines a colormap.
    %   The colors begin with bright blue, range through shades of
    %   blue to white, and then through shades of red to bright red.
    %   REDBLUE, by itself, is the same length as the current figure's
    %   colormap. If no figure exists, MATLAB creates one.
    %
    %   For example, to reset the colormap of the current figure:
    %
    %             colormap(redblue)
    %
    %   See also HSV, GRAY, HOT, BONE, COPPER, PINK, FLAG, 
    %   COLORMAP, RGBPLOT.
    %   Adam Auton, 9th October 2009
    if nargin < 1, m = size(get(gcf,'colormap'),1); end
    if (mod(m,2) == 0)
        % From [0 0 1] to [1 1 1], then [1 1 1] to [1 0 0];
        m1 = m*0.5;
        r = (0:m1-1)'/max(m1-1,1);
        g = r;
        r = [r; ones(m1,1)];
        g = [g; flipud(g)];
        b = flipud(r);
    else
        % From [0 0 1] to [1 1 1] to [1 0 0];
        m1 = floor(m*0.5);
        r = (0:m1-1)'/max(m1,1);
        g = r;
        r = [r; ones(m1+1,1)];
        g = [g; 1; flipud(g)];
        b = flipud(r);
    end
    c = [r g b]; 
end    