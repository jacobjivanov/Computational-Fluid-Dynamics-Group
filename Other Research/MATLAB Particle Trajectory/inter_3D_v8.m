function inter_zyx_value = inter_3D_v8(x_coords, y_coords, z_coords, values3D, pos)
   Ni = size(values3D, 1);
   Nj = size(values3D, 2);
   % Nk = size(values3D, 3);

   inter_zy_values = zeros(1, Ni);
   for i = 1:Ni
       inter_y_values = zeros(1, Nj);
       for j = 1:Nj
          inter_y_values(j) = inter_1D_v8(z_coords, values3D(i, j, :), pos(3));
       end
       inter_zy_values(i) = inter_1D_v8(y_coords, inter_y_values, pos(2));
   end

   inter_zyx_value = inter_1D_v8(x_coords, inter_zy_values, pos(1));
end