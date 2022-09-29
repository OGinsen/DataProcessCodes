import numpy as np
import os

'''
Some functions to divide active segment data into window data by sliding window
'''

def Sliding_windows(data,window_size,step_size):
	all_windows_data = []
	for i in range(0,len(data)-window_size,step_size):
		window_data = data[i:i+window_size,:]
		all_windows_data.append(window_data)
	return all_windows_data

def Save_windows_data(path):
	relax_num = 0
	tired_num = 0
	for root, dirs, files in os.walk(path,'r'):
		for i in range(len(files)):
			if (files[-3=='txt'])and(files[i][5:10]=='relax'):
				file_path = root + '\\' + files[i]
				emg = np.loadtxt(file_path,delimiter=',',skiprows=0)
				windows_data = Sliding_windows(emg,1024,512)
				for win_data in windows_data:
					np.savetxt('D:\\All_Datasets\\ActiveDatasetsWindows\\' + file_path[31:40] + file_path[31:33] + '_windows_' + file_path[45:53]
							   +str(relax_num)+'.csv',win_data,delimiter=',')
					relax_num += 1
			if (files[-3=='txt'])and(files[i][5:10]=='tired'):
				file_path = root + '\\' + files[i]
				emg = np.loadtxt(file_path,delimiter=',',skiprows=0)
				windows_data = Sliding_windows(emg,1024,512)
				for win_data in windows_data:
					np.savetxt('D:\\All_Datasets\\ActiveDatasetsWindows\\' + file_path[31:40] + file_path[31:33] + '_windows_' + file_path[45:53]
							   +str(tired_num)+'.csv',win_data,delimiter=',')
					tired_num += 1