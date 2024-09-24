function inter_yx_value = inter_2D(x_coords, y_coords, values2D, pos)
   Ni = size(values2D, 1);
   % Nj = size(values2D, 2); % Not actually necessary

   inter_y_values = zeros(1, Ni);
   for i = 1:Ni
       inter_y_values(i) = inter_1D(y_coords, values2D(i, :), pos(2));
   end

   inter_yx_value = inter_1D(x_coords, inter_y_values, pos(1));
end