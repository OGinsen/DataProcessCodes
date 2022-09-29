import numpy as np
from ActiveSegmentDetection.ShortEnergyFun import Active_Segment_Detection,Average_Short_time_energy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',size = 35)
plt.rcParams['font.sans-serif'] = ['Times New Roman']

'''
Display active segment interception and short-term energy
'''

def Show_active_segment(path,th1,th2):
	data = np.loadtxt(path,delimiter=',',skiprows=0)
	E, x, y, raw_x, raw_y = Active_Segment_Detection(data, 64, 32,
													 th1=th1,
													 th2=th2,
													 channel=0)
	if (path.split('\\')[4]=='relax'):
		num = []
		for i, j in enumerate(y):
			if j == np.max(E) * 0.6:
				num.append(i)
		number = len(num) / 2
		print('Number of active segments for non fatigue data:', number)
		plt.figure(1,figsize=(10,8))
		plt.title('Short time energy distribution of non fatigue data',fontsize=24)
		plt.plot(E, marker='.',label='Short time energy distribution')
		plt.plot(x, y, 'r', marker='.',label = 'Active segments')
		plt.ylabel('Short time energy value',fontsize=24)
		plt.xlabel('Sampling point',fontsize=24)
		plt.legend(fontsize=24)
		plt.tick_params(labelsize=24)
		plt.grid(linestyle='--')
		plt.figure(2,figsize=(10,8))
		plt.title('Active segment detection of non fatigue data',fontsize=24)
		plt.plot(data[:, 0], label='Raw NF-signal')
		plt.plot(raw_x, raw_y, 'r', marker='.',label = 'Active segments')
		plt.ylabel('Normalized amplitude',fontsize=24)
		plt.xlabel('Sampling point',fontsize=24)
		plt.legend(fontsize=24)
		plt.tick_params(labelsize=24)
		plt.grid(linestyle='--')
		plt.show()
	elif (path.split('\\')[4] == 'tired'):
		num = []
		for i, j in enumerate(y):
			if j == np.max(E) * 0.6:
				num.append(i)
		number = len(num) / 2
		print('Number of active segments for fatigue data:', number)
		plt.figure(3,figsize=(10,8))
		plt.title('Short time energy distribution of fatigue data', fontsize=24)
		plt.plot(E, marker='.',label='Short time energy distribution')
		plt.plot(x, y, 'r', marker='.',label = 'Active segments')
		plt.ylabel('Short time energy value',fontsize=24)
		plt.xlabel('Sampling point',fontsize=24)
		plt.legend(fontsize=24)
		plt.tick_params(labelsize=24)
		plt.grid(linestyle='--')
		plt.figure(4,figsize=(10,8))
		plt.title('Active segment detection of fatigue data', fontsize=24)
		plt.plot(data[:, 0], label='Raw F-signal')
		plt.plot(raw_x, raw_y, 'r', marker='.',label = 'Active segments')
		plt.ylabel('Normalized amplitude',fontsize=24)
		plt.xlabel('Sampling point',fontsize=24)
		plt.legend(fontsize=24)
		plt.tick_params(labelsize=24)
		plt.grid(linestyle='--')
		plt.show()

if __name__ == "__main__":
	relax_data_path = r'D:\All_Datasets\Datasets_Norm\S1\relax\norm_relax_0.txt'
	tired_data_path = r'D:\All_Datasets\Datasets_Norm\S1\tired\norm_tired_0.txt'
	Show_active_segment(relax_data_path,6.91301041466369E-06,0.000026815563718975)
	Show_active_segment(tired_data_path, 0.000008463464834161, 7.67684218755959E-06)