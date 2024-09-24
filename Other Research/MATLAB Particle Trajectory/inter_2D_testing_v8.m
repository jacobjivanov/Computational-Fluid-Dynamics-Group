% inter_2D_testing_v8

grid = 100;
domain_max = 2 * pi;

x = linspace(0, domain_max, grid + 1); x(end) = [];
y = linspace(0, domain_max, grid + 1); y(end) = [];

values2D = zeros(grid, grid);
for i = 1:grid
    for j = 1:grid
        values2D(i, j) = exp(sin(x(i)) + sin(y(j)));
    end
end

grid_inter = grid * 4;

x_inter = linspace(0, domain_max, grid_inter + 1); x_inter(end) = [];
y_inter = linspace(0, domain_max, grid_inter + 1); y_inter(end) = [];

values2D_inter = zeros(grid_inter, grid_inter);
values2D_exact = zeros(grid_inter, grid_inter);

for i = 1:grid_inter
    for j = 1:grid_inter
        values2D_inter(i, j) = inter_2D_v8(x, y, values2D, [x_inter(i), y_inter(j)]);
        values2D_exact(i, j) = exp(sin(x_inter(i)) + sin(y_inter(j)));
    end
end

values2D_error = values2D_inter - values2D_exact;
% pcolor(x, y, values2D)
% pcolor(x_inter, y_inter, values2D_exact)
pcolor(x_inter, y_inter, values2D_error); colorbar