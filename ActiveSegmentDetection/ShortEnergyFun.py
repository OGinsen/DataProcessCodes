import numpy as np
from scipy import signal
from scipy.signal.signaltools import *
from scipy.fftpack import fft,ifft

'''
Intercept function of short-time energy activity segment
'''

def Average_Short_time_energy(data):
	square = np.square(data)
	sum = np.sum(square)
	Average_energy = sum/(data.shape[0]*data.shape[1])
	return Average_energy

def Active_Segment_Detection(data,window_size,step_size,th1,th2,channel):
	E = []
	TH1 = th1
	TH2 = th2
	for index in range(0,data.shape[0],step_size):
		e = Average_Short_time_energy(data[index: index + window_size,:])
		E.append(e)
	start_end = []
	found = False
	for i in range(len(E) - 3):
		E_e = E[i:i + 3]
		if not found:
			if all(e_value > TH1 for e_value in E_e):
				index_start = i
				found = True
			else:
				continue
		else:
			if all(e_value < TH2 for e_value in E_e):
				index_end = i
				found = False
				start_end.append((index_start,index_end))
	y = []
	x = []
	raw_x = []
	raw_y = []
	D_min = (2.5* 2000) / step_size
	D_max = (3.9 * 2000) / step_size
	TH4 = 0.00001
	for i in range(len(start_end)):
		if start_end[i][1]-start_end[i][0]>=D_min:
			Average_e = np.sum(np.square(data[start_end[i][0]*step_size:start_end[i][1]*step_size,:]))/len(
										 data[start_end[i][0]*step_size:start_end[i][1]*step_size,:])
			if Average_e > TH4:
				y.extend((0,np.max(E)*0.6,np.max(E)*0.6,0))
				x.extend([start_end[i][0]]*2)
				x.extend([start_end[i][1]]*2)
				raw_y.extend((0,np.max(data[:,channel])*0.6,np.max(data[:,channel])*0.6,0))
				raw_x.extend([start_end[i][0] *step_size] * 2)
				raw_x.extend([start_end[i][1] *step_size] * 2)
	return E, x, y, raw_x, raw_y


