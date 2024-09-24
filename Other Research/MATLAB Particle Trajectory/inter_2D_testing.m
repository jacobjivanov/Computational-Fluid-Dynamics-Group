% test2
test = 1;
if test == 1
    Ni = 30;
    Nj = 30;
    x_max = 100 * rand();
    y_max = 100 * rand();
    x = linspace(0, x_max, Ni);
    y = linspace(0, y_max, Nj);
    
    values2D = zeros(Ni, Nj);
    norm_x = x .* 2 * pi ./ x_max;
    norm_y = y .* 2 * pi ./ y_max;
    for i = 1:Ni
        for j = 1:Nj
            values2D(i, j) = exp(sin(norm_x(i)) + sin(norm_y(j)));
        end
    end

    % values2D(11, 18) CHECKS OUT
    Ni_inter = Ni * 4;
    Nj_inter = Nj * 4;

    x_inter = linspace(0, x_max, Ni_inter);
    y_inter = linspace(0, y_max, Nj_inter);
    
    values2D_inter = zeros(Ni_inter, Nj_inter);
    values2D_exact = zeros(Ni_inter, Nj_inter);
    values2D_error = zeros(Ni_inter, Nj_inter);

    norm_x = x_inter .* 2 * pi ./ x_max;
    norm_y = y_inter .* 2 * pi ./ y_max;
    for i = 1:Ni_inter
        for j = 1:Nj_inter
            values2D_inter(i, j) = inter_2D(x, y, values2D, [x_inter(i), y_inter(j)]);
            values2D_exact(i, j) = exp(sin(norm_x(i)) + sin(norm_y(j)));
        end
    end

    values2D_error = values2D_inter - values2D_exact;
    % pcolor(x, y, values2D)
    pcolor(x_inter, y_inter, values2D_error); colorbar
end

if test == 2
    x = linspace(0, 2 * pi, 14);
    y = [0.79379, 0.65885, 0.26912, 0.60397, 0.12430, 0.07876, 0.51430, 0.74956, 0.28334, 0.66792, 0.35352, 0.04800, 0.01439, 0.12871];
    
    pos = 2.1;
    x_i = inter_1D(x, y, pos)
    
    plot(x, y)
end

if test == 3
    x = linspace(0, 2 * pi, 30);
    y = [1.00000000000000, 1.23982524745171, 1.52179278181533, 1.83157126863590, 2.14290429283459, 2.41938177933111, 2.62098445274053, 2.71429815589422, 2.68270743774812, 2.53191699109498, 2.28802499571134, 1.98913418044017, 1.67456571524170, 1.37616621793835, 1.11418034548409, 0.897520768565990, 0.726656407463710, 0.597169756252693, 0.502731293762553, 0.437058162334062, 0.394957656004169, 0.372757754322776, 0.368419363889135, 0.381536028935383, 0.413328730729084, 0.466656398675286, 0.545979300464116, 0.657119689322690, 0.806565281724472, 1.00000000000000];
    
    pos = 2.1;
    x_i = inter_1D(x, y, pos)
    
    plot(x, y)
end

if test == 4
    x = linspace(0, 2 * pi, 30);
    y = [0.4613, 0.6361, 0.2637, 0.0480, 0.4013, 0.1011, 0.1270, 0.8851, 0.5284, 0.0644, 0.6606, 0.5225, 0.2979, 0.6718, 0.2959, 0.1857, 0.5637, 0.8624, 0.4132, 0.9710, 0.0605, 0.5345, 0.5416, 0.0966, 0.2570, 0.2175, 0.7798, 0.7260, 0.4283, 0.3716]
    pos = 2.12;
    x_i = inter_1D(x, y, pos)
    
    plot(x, y)
end

