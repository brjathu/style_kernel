import os


range_sigma = [1e18, 1e19, 1e21, 1e22, 1e25, 1e30]

os.system("mkdir final")
os.system("mkdir final/dot")
os.system("mkdir final/exp")
for sig in range_sigma:
    os.system("mkdir final/exp/" + str(sig))
