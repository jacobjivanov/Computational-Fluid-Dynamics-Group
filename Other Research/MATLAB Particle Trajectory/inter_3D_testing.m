% inter_2D_testing_v8

grid = 12;
domain_max = 2 * pi;

x = linspace(0, domain_max, grid + 1); x(end) = [];
y = linspace(0, domain_max, grid + 1); y(end) = [];
z = linspace(0, domain_max, grid + 1); z(end) = [];


values3D = zeros(grid, grid, grid);
for i = 1:grid
    for j = 1:grid
        for k = 1:grid
            values3D(i, j, k) = exp(sin(x(i)) + sin(y(j)) + sin(z(k)));
        end
    end
end

grid_inter = grid * 4;

x_inter = linspace(0, domain_max, grid_inter + 1); x_inter(end) = [];
y_inter = linspace(0, domain_max, grid_inter + 1); y_inter(end) = [];
z_inter = linspace(0, domain_max, grid_inter + 1); z_inter(end) = [];

values3D_inter = zeros(grid_inter, grid_inter, grid_inter);
values3D_exact = zeros(grid_inter, grid_inter, grid_inter);

%{
for i = 1:grid_inter
    for j = 1:grid_inter
        for k = 1:grid_inter
            values3D_inter(i, j, k) = inter_3D_v8(x, y, z, values3D, ...
                [x_inter(i), y_inter(j), z_inter(k)]);
            values3D_exact(i, j, k) = exp(sin(x_inter(i)) + ...
                sin(y_inter(j)) + sin(z_inter(k)));
        end
    end
end

values3D_error = values3D_inter - values3D_exact;

s = 0;
for i = 1:grid_inter
    for j = 1:grid_inter
        for k = 1:grid_inter
            s = s + values3D_error(i, j, k)^2;
        end
    end
end
%}

a = inter_3D_v8(x, y, z, values3D, ...
                [1, 1.2, 2.9])

x
y
z
values3D(1, :, :)
% pcolor(x, y, values2D)
% pcolor(x, y, values3D(:, :, 1))
pcolor(x_inter, y_inter, values3D_inter(:, :, 6)); colorbar