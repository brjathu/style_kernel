import os


range_sigma = [15000, 18000, 19000, 20000, 21000, 22000, 25000, 30000]

os.system("mkdir final")
os.system("mkdir final/dot")
os.system("mkdir final/exp")
for sig in range_sigma:
    os.system("mkdir final/exp/" + str(sig))
