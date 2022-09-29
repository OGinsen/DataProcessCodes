import numpy as np
from scipy import signal
from scipy.signal.signaltools import filtfilt
from Denoise.DenoiseFun import NLMS

'''
Save active segment and filter
'''

def Butterworth_Filter(Raw_1):
	b,a = signal.butter(8,[0.020,0.500],'bandpass')  #  #Bandpass filter, Passband5-200HZ,0.01=5/(sampling_rate/2),0.4=200/(sampling_rate/2), 3rd order bandpass
	raw_1 = filtfilt(b,a,Raw_1)
	return raw_1



def Save_and_Filtering_Active_segment(list_x,list_y,path,data,rest):
	#Firstly, the active segment is intercepted, and then the adaptive filtering NLMS is used
	start_end = []
	Coordinate_set = list(zip(list_x, list_y))
	for i in range(len(Coordinate_set)):
		if Coordinate_set[i][1] > 0:
			start_end.append(Coordinate_set[i][0])
	for i in range(0, len(start_end), 2):
		if path[-11:-6] == 'relax':
			Active_segment = data[start_end[i]:start_end[i + 1], :]
			for c in range(Active_segment.shape[1]):
				[yn,Wn,en] = NLMS(rest[:len(Active_segment[:,c]),c],Active_segment[:,c],64,0.03,len(Active_segment[:,c]))
				Active_segment[:,c] = en
				Active_segment[:, c] = Butterworth_Filter(Active_segment[:, c]) ##The components below 20Hz and above 500Hz are filtered by band-pass filter
			np.savetxt('D:\\All_Datasets\\ActiveDatasets\\' + path[-25:-4] + '_' + str(int(i / 2)) + '.txt', Active_segment, delimiter=',')
		elif path[-11:-6] == 'tired':
			Active_segment = data[start_end[i]:start_end[i + 1], :]
			for c in range(4):
				[yn, Wn, en] = NLMS(rest[:len(Active_segment[:, c]), c], Active_segment[:, c], 64, 0.03,len(Active_segment[:, c]))
				Active_segment[:, c] = en
				Active_segment[:, c] = Butterworth_Filter(Active_segment[:, c])  ##The components below 20Hz and above 500Hz are filtered by band-pass filter
			np.savetxt('D:\\All_Datasets\\ActiveDatasets\\' + path[-25:-4] + '_' + str(int(i / 2)) + '.txt', Active_segment, delimiter=',')




