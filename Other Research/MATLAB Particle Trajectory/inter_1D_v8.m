function inter_x_value = inter_1D_v8(x_coords, values1D, pos)
   values1D_fft = fft(values1D);

   Ni = size(values1D, 2);
   x_max = x_coords(end) + x_coords(2);

   inter_x_value = real(values1D_fft(1));
   for f = 2:floor(Ni / 2) + 1
       w = (f - 1) * 2 * pi / x_max;
       inter_x_value = inter_x_value + 2 * real(values1D_fft(f)) * cos(w * pos);
       inter_x_value = inter_x_value - 2 * imag(values1D_fft(f)) * sin(w * pos);
   end
   inter_x_value = inter_x_value / Ni;