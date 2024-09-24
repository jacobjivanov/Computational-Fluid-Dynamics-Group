import numpy as np

def process(Ni):
   Ni_inter = 4 * Ni
   Nj_inter, Nk_inter = Ni_inter, Ni_inter

   error_fi6 = np.load("Convergence Test Outputs/error_fi6[Ni = {0}].npy".format(Ni))
   error_li6 = np.load("Convergence Test Outputs/error_li6[Ni = {0}].npy".format(Ni))
   
   # Arrays from array_generation.py are correctly importing
   # print(error_fi6[2, 3, 4])

   error_fi6_flat = np.ravel(error_fi6)
   error_li6_flat = np.ravel(error_li6)

   error_fi6_max = np.max(error_fi6_flat)
   error_li6_max = np.max(error_li6_flat)

   def l2_norm(error_flat, N):
      s = 0
      x_max = 2 * np.pi
      for e in error_flat:
         s += np.abs(e) ** 2 * (x_max  ** 3) / ((N * 4 - 1) ** 3)
      return np.sqrt(s)

   error_fi6_l2 = l2_norm(error_fi6_flat, Ni)
   error_li6_l2 = l2_norm(error_li6_flat, Ni)
   
   print("L2 error_fi6: {0}".format(error_fi6_l2))
   print("L2 error_li6: {0}".format(error_li6_l2))
   print("Max error_fi6: {0}".format(error_fi6_max))
   print("Max error_li6: {0}".format(error_li6_max))

   import matplotlib.pyplot as plt
   plt.hist(np.log10(np.abs(error_fi6_flat) + 1e-15), bins = 100, alpha = 0.5, color = 'blue', density = True, label = "Trifourier Error")
   plt.hist(np.log10(np.abs(error_li6_flat) + 1e-15), bins = 100, alpha = 0.5, color = 'red', density = True, label = "Trilinear Error")
   plt.xlim(-20, 5)
   plt.xlabel(r"$\log_{10} \left[ |e| + 10^{-15}\right]$")
   plt.ylabel("Density")
   plt.legend()
   plt.title("Density Distribution of Error, $N = ${0}".format(Ni))

   plt.show()

process(12)