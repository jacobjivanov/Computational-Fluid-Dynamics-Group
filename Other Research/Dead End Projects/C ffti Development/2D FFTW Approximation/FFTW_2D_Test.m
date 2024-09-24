t = linspace(1, 200, 200);
[x, y] = meshgrid(t);

signal2D = log(x) + sin(y);

signal2D_fft = fft2(signal2D);
signal2D_approx = ifft2(signal2D_fft);

surf(signal2D - signal2D_approx)
