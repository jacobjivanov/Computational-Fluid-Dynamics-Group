t = linspace(1, 10, 10);
[x, y] = meshgrid(t);

signal2D = log(x) + sin(y);
signal2D_size = size(signal2D);

m = signal2D_size(1);
n = signal2D_size(2);

signal2D_fft = fft2(signal2D);

% Wavenumber functions determined by referencing documentation found below
% https://www.mathworks.com/help/matlab/ref/fft2.html
omega_m = @(m) exp(-2 * pi * 1i / m);
omega_n = @(n) exp(-2 * pi * 1i / n);

signal2D_approx = zeros(size(signal2D));

for j = 0:m - 1
    for k = 0:n - 1
        a = 
        signal2D_approx(j + 1, k + 1) = signal2D_approx(j + 1, k + 1) + a;
    end
end

surf(signal2D)
surf(signal2D_approx)