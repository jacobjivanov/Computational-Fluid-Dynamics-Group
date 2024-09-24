import numpy as np
import matplotlib.pyplot as plt

Ni = np.array([4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])

form1 = np.array([0.4686623466879783, 0.24840503462471422, 0.1347173667254589, 0.01847147467838594, 0.0014983824428438247, 0.0012547467065611017, 0.0007601291392130287, 7.685824066199083e-05, 4.7939725525769715e-06, 4.251599205952112e-06, 2.6952581726807745e-06, 2.1356647389481817e-07, 1.0737154895245656e-08, 9.803830894491752e-09, 6.370667606295399e-09, 4.138387447978218e-10, 1.738207275446116e-11, 1.6155389781844026e-11, 1.0662858050428396e-11, 5.862826771195704e-13, 2.1134600328979038e-14, 1.987421481116218e-14, 1.32552888843613e-14, 6.664709988187903e-16, 2.0563075373634527e-16, 2.272312873823341e-16, 2.1387445961315464e-16, 1.9689110811140152e-16, 1.7496528721238836e-16, 1.7258845598208168e-16, 1.5772840771034928e-16, 1.4742852387460732e-16, 1.4090357402083365e-16, 1.5621850651782811e-16, 1.5074590228852336e-16, 1.4166544059101254e-16])

form2 = np.array([0.0888148726690935, 0.06292728941236046, 0.034109783254395745, 0.004678283485597188, 0.0003792001105844164, 0.0003177895434032107, 0.0001925177841299499, 1.9465876490796206e-05, 1.2141686107673771e-06, 1.0768019707560914e-06, 6.826276822857969e-07, 5.408995271107439e-08, 2.719397794861712e-09, 2.483014960468639e-09, 1.6134981537492475e-09, 1.0481287533866079e-10, 4.402344365205191e-12])

form3 = np.array([3.055881016729239, 2.5440741293985205, 1.7317412160700503, 1.4614244634355222, 1.1586073694003403, 0.9920532835397474, 0.814103031985956, 0.7237255320821528, 0.6050458027809157, 0.5367757579873423, 0.4806007980003212, 0.4380447655836663, 0.37616024004593007, 0.3446937733707412, 0.31755964347520377, 0.2843665908858675, 0.2589537405480047])

fig, ax = plt.subplots(1, 1)

ax.scatter(Ni, form1, label = r"$u = \exp \left[ \sin(x) + \sin(y) + \sin(z) \right]$", color = 'blue')
ax.scatter(Ni[:len(form2)], form2, label = r"$u = \exp \left[ \sin(x) \right] + \sin(y) + \sin(z)$", color = 'red')
ax.scatter(Ni[:len(form3)], form3, label = r"$u = \exp \left[ \sin(x) - \cos(yz) \right]$", color = 'green')
ax.axhline(2.22044604925e-16, label = "64 Bit Precision", linestyle = 'dashed', color = 'gray')
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_title("Error Convergence of inter_3D()")
ax.set_xlabel("Ni")
ax.set_ylabel("2-Norm of Signed Interpolation Error")
ax.set_ylim(1e-16, 1e2)
ax.legend(loc = "lower left")

# plt.show()
fig.savefig('inter_3D() Error Convergence', dpi = 600)