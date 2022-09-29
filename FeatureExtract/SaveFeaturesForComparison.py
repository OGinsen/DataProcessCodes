import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fftpack import fft
import pandas as pd
from Denoise.DenoiseFun import fft_Spectrum_Map,Power_spectrum
from FeatureExtract.FeatureExtractFun import Feature_iEMG,Feature_RMS,Feature_MPF,Feature_MF
plt.rcParams['font.sans-serif'] = ['Times New Roman']



# path = 'D:\\All_Datasets\\ActiveDatasetsWindowsFeatureShow\\'
path = 'D:\\All_Datasets\\ActiveDatasets\\S1\\'
for G in range(7):
	iEMG_r, RMS_r, MPF_r, MF_r = [], [], [], []
	iEMG_t, RMS_t, MPF_t, MF_t = [], [], [], []
	for root, dirs, files in os.walk(path,'r'):
		print('root:',root)
		print('dirs:',dirs)
		print('files:',files)
		for i in range(len(files)):
			# if (files[i][8:15]=='relax_0'):
			if (files[i][5:12] == 'relax_' + str(G)):
				file_path = root + '\\' + files[i]
				print(file_path)
				window_emg = np.loadtxt(file_path,delimiter=',',skiprows=0)
				iEMG = Feature_iEMG(window_emg)
				RMS = Feature_RMS(window_emg)
				MPF, MF  = [], []
				for ch in range(4):
					f, p = Power_spectrum(window_emg[:,ch],len(window_emg[:,ch]),2000)
					MPF.append(Feature_MPF(f,p))
					MF.append(Feature_MF(f,p))
				MPF, MF = np.array(MPF),np.array(MF)
				iEMG_r.append(iEMG)
				RMS_r.append(RMS)
				MPF_r.append(MPF)
				MF_r.append(MF)
			elif (files[i][5:12] == 'tired_' + str(G)):
				file_path = root + '\\' + files[i]
				print(file_path)
				window_emg = np.loadtxt(file_path,delimiter=',',skiprows=0)
				iEMG = Feature_iEMG(window_emg)
				RMS = Feature_RMS(window_emg)
				MPF, MF  = [], []
				for ch in range(4):
					f, p = Power_spectrum(window_emg[:,ch],len(window_emg[:,ch]),2000)
					MPF.append(Feature_MPF(f,p))
					MF.append(Feature_MF(f,p))
				MPF, MF = np.array(MPF),np.array(MF)
				iEMG_t.append(iEMG)
				RMS_t.append(RMS)
				MPF_t.append(MPF)
				MF_t.append(MF)
	iEMG_r,RMS_r,MPF_r,MF_r = np.array(iEMG_r),np.array(RMS_r),np.array(MPF_r),np.array(MF_r)
	iEMG_t,RMS_t,MPF_t,MF_t = np.array(iEMG_t),np.array(RMS_t),np.array(MPF_t),np.array(MF_t)
	print(iEMG_r.shape)
	np.savetxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'iEMG_r_G' + str(G)+ '.csv',iEMG_r,delimiter=',')
	np.savetxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'RMS_r_G' + str(G)+ '.csv',RMS_r,delimiter=',')
	np.savetxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MPF_r_G' + str(G)+ '.csv',MPF_r,delimiter=',')
	np.savetxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MF_r_G' + str(G)+ '.csv',MF_r,delimiter=',')
	np.savetxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'iEMG_t_G' + str(G)+ '.csv',iEMG_t,delimiter=',')
	np.savetxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'RMS_t_G' + str(G)+ '.csv',RMS_t,delimiter=',')
	np.savetxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MPF_t_G' + str(G)+ '.csv',MPF_t,delimiter=',')
	np.savetxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MF_t_G' + str(G)+ '.csv',MF_t,delimiter=',')