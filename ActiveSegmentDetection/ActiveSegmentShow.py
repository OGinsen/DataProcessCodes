import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']

'''
Display the intercepted active segment signal
'''

path = 'D:\\All_Datasets\\ActiveDatasets\\'

for root, dirs, files in os.walk(path ,'r'):
	for i in range(len(files)):
		if (files[-3 == 'txt']):
			file_path = root + '\\' + files[i]
			emg = np.loadtxt(file_path, delimiter=',', skiprows=0)
			plt.figure(1)
			plt.suptitle('Active segment data',fontsize=20)
			plt.plot(emg[:,0],label='Active segment Ch0')
			plt.ylabel('Normalized amplitude', fontsize=15)
			plt.xlabel('Sampling point', fontsize=15)
			plt.legend(fontsize=15)
			plt.grid()
			plt.show()