import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import thermophysical_properties as tp

T_inter = np.linspace(200, 2500, 100)

k_BeO_inter = np.zeros(100)
k_Steel_inter = np.zeros(100)

cP_BeO_inter = np.zeros(100)
cP_Steel_inter = np.zeros(100)

rho_BeO = np.zeros(100)
rho_Steel = np.zeros(100)

alpha_BeO_inter = np.zeros(100)
alpha_Steel_inter = np.zeros(100)

for i in range(len(T_inter)):
   k_BeO_inter[i] = tp.k_BeO_fit(T_inter[i], tp.k_BeO_params[0], tp.k_BeO_params[1])
   k_Steel_inter[i] = tp.k_Steel_fit(T_inter[i], tp.k_Steel_params[0], tp.k_Steel_params[1])

   cP_BeO_inter[i] = tp.cP_BeO_fit(T_inter[i], tp.cP_BeO_params[0], tp.cP_BeO_params[1])
   cP_Steel_inter[i] = tp.cP_Steel_fit(T_inter[i], tp.cP_Steel_params[0], tp.cP_Steel_params[1])

   rho_BeO[i] = tp.rho(0, 0)
   rho_Steel[i] = tp.rho(0.020, 0)

   alpha_BeO_inter[i] = k_BeO_inter[i] / (cP_BeO_inter[i] * rho_BeO[i])
   alpha_Steel_inter[i] = k_Steel_inter[i] / (cP_Steel_inter[i] * rho_Steel[i])

chart = "alpha" # change to whatever you want
if chart == "cP":
   plt.title("Thermophysical Properties over Temperature:\nIsobaric Specific Heat")
   plt.ylabel(r"Isobaric Specific Heat $\left(\frac{\mathrm{J}}{\mathrm{kg \cdot K}}\right)$")
   plt.xlabel("Temperature (K)")
   plt.xlim(0, 2500) # K
   plt.ylim(0, 3000) # J / kg K

   plt.scatter(tp.T_Steel, tp.cP_Steel, color = "blue")
   plt.plot(T_inter, cP_Steel_inter, color = "blue", linestyle = "dashed", label = "AISA 304 Steel")

   plt.scatter(tp.T_BeO, tp.cP_BeO, color = "red")
   plt.plot(T_inter, cP_BeO_inter, color = "red", linestyle = "dashed", label = "Beryllium Oxide")
   plt.legend()
   plt.savefig("Isobaric Specific Heat.png", dpi = 300)
   plt.show()

if chart == "k":
   plt.title("Thermophysical Properties over Temperature:\nThermal Conductivity")
   plt.ylabel(r"Thermal Conductivity $\left(\frac{\mathrm{W}}{\mathrm{m \cdot K}}\right)$")
   plt.xlabel("Temperature (K)")
   plt.xlim(0, 2500) # K
   plt.ylim(0, 500) # W / m K

   plt.scatter(tp.T_Steel, tp.k_Steel, color = "blue")
   plt.plot(T_inter, k_Steel_inter, color = "blue", linestyle = "dashed", label = "AISA 304 Steel")

   plt.scatter(tp.T_BeO, tp.k_BeO, color = "red")
   plt.plot(T_inter, k_BeO_inter, color = "red", linestyle = "dashed", label = "Beryllium Oxide")
   plt.legend()
   plt.savefig("Thermal Conductivity.png", dpi = 300)
   plt.show()

if chart == "rho":
   plt.subplots_adjust(left = 0.15)
   plt.title("Thermophysical Properties over Temperature:\nDensity")
   plt.ylabel(r"Density $\left(\frac{\mathrm{kg}}{\mathrm{m}^3}\right)$")
   plt.xlabel("Temperature (K)")
   plt.xlim(0, 2500) # K
   plt.ylim(0, 10000) # kg / m3

   plt.plot(T_inter, rho_Steel, color = "blue", linestyle = "dashed", label = "AISA 304 Steel")
   plt.plot(T_inter, rho_BeO, color = "red", linestyle = "dashed", label = "Beryllium Oxide")
   plt.legend()
   plt.savefig("Density.png", dpi = 300)
   plt.show()

if chart == "alpha":
   plt.subplots_adjust(left = 0.15)
   plt.title("Thermophysical Properties over Temperature:\nThermal Diffusivity")
   plt.ylabel(r"Thermal Diffusivity $\left(\frac{\mathrm{m}^2}{\mathrm{s}}\right)$")
   plt.xlabel("Temperature (K)")
   plt.xlim(0, 2500) # K
   plt.ylim(0, 10e-5) # m2 / s

   plt.plot(T_inter, alpha_Steel_inter, color = "blue", linestyle = "dashed", label = "AISA 304 Steel")

   plt.plot(T_inter, alpha_BeO_inter, color = "red", linestyle = "dashed", label = "Beryllium Oxide")
   plt.legend()
   plt.savefig("Thermal Diffusivity.png", dpi = 300)
   plt.show()