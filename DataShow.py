import numpy as np
import matplotlib
matplotlib.rc('font',size = 24)
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']

path = r'D:\All_Datasets\Datasets_Norm\S1\relax\norm_relax_0.txt'
# path = r'D:\All_Datasets\Datasets_Norm\S8\tired\norm_tired_0.txt'
emg = np.loadtxt(path,delimiter=',',skiprows=0)
plt.figure(figsize=(10,7))
plt.plot(emg[:,0])
# plt.title('',fontsize=20)
plt.xlabel('Sampling point',fontsize=24)
plt.ylabel('Normalized amplitude',fontsize=24)


plt.tick_params(labelsize=24)
plt.grid(linestyle='--')
plt.show()
