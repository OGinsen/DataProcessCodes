import os
import pandas as pd
from Denoise.DenoiseFun import Power_spectrum
from FeatureExtract.FeatureExtractFun import *
import time
np.set_printoptions(threshold=np.inf)





def FeatureExtractSave(data_path,datasets_type,features_save_psth):
	for root,dirs,files in os.walk(data_path,'r'):
		start = time.time()
		result = np.zeros((1,209))
		for i in range(len(files)):
			if (files[i][11:16] == datasets_type):
				file_path = root + '\\' + files[i]
				print('file_path:',file_path)
				window_data = np.loadtxt(file_path, delimiter=',', skiprows=0)
				# Time domain feature extraction
				iEMG = Feature_iEMG(window_data)
				RMS = Feature_RMS(window_data)
				MAV = Feature_MAV(window_data)
				MAV1 = Feature_MAV1(window_data)
				MAV2 = Feature_MAV2(window_data)
				VAR = Feature_VAR(window_data)
				SSI = Feature_SSI(window_data)
				VOrder = Feature_VOrder(window_data)
				DASDV = Feature_DASDV(window_data)
				MAX = Feature_MAX(window_data)
				MIN = Feature_MIN(window_data)
				Range = Feature_range(window_data)
				LOG = Feature_LOG(window_data)
				WL = Feature_WL(window_data)
				AAC = Feature_AAC(window_data)
				MFL = Feature_MFL(window_data)
				WAMP = Feature_WAMP(window_data,th=1e-4)
				ZC = Feature_ZC(window_data,th=1e-4)
				SSC = Feature_SSC(window_data,th=-1e-8)
				Sk1,Sk2,Sk3,Sk4 = Feature_SK(window_data)
				Kr1,Kr2,Kr3,Kr4 = Feature_Kr(window_data)
				# Frequency domain feature extraction
				MPF, MF, PKF, MNP, TTP = [], [], [], [], []
				# Time-frequency domain feature extraction
				MOAC_1, MOAC_2, MOAC_3, MOAC_4 = [], [], [], []
				APOC_1, APOC_2, APOC_3, APOC_4 = [], [], [], []
				STDOC_1, STDOC_2, STDOC_3, STDOC_4 = [], [], [], []
				R_1, R_2, R_3 = [], [], []
				for ch in range(4):
					# Frequency domain characteristic calculation
					f , p = Power_spectrum(window_data[:,ch],len(window_data[:,ch]),2000)
					MPF.append(Feature_MPF(f,p))
					MF.append(Feature_MF(f, p))
					PKF.append(Feature_PKF(p))
					MNP.append(Feature_MNP(p))
					TTP.append(Feature_TTP(p))
					# Time frequency domain characteristic calculation
					M_1, M_2, M_3, M_4 = Feature_MOAC(window_data[:,ch])
					MOAC_1.append(M_1)
					MOAC_2.append(M_2)
					MOAC_3.append(M_3)
					MOAC_4.append(M_4)
					A_1, A_2, A_3, A_4 = Feature_APOC(window_data[:, ch])
					APOC_1.append(A_1)
					APOC_2.append(A_2)
					APOC_3.append(A_3)
					APOC_4.append(A_4)
					S_1, S_2, S_3, S_4 = Feature_STDOC(window_data[:, ch])
					STDOC_1.append(S_1)
					STDOC_2.append(S_2)
					STDOC_3.append(S_3)
					STDOC_4.append(S_4)
					_1, _2, _3 = Feature_R(window_data[:, ch])
					R_1.append(_1)
					R_2.append(_2)
					R_3.append(_3)
				# Feature extraction of sEMG raw signal based on parameter model
				AR = Feature_AR(window_data)
				Beta   = AR[0, :]
				alpha1 = AR[1, :]
				alpha2 = AR[2, :]
				alpha3 = AR[3, :]
				alpha4 = AR[4, :]
				MPF, MF, PKF, MNP, TTP = np.array(MPF),np.array(MF),np.array(PKF),np.array(MNP),np.array(TTP)
				MOAC_1, MOAC_2, MOAC_3, MOAC_4 = np.array(MOAC_1),np.array(MOAC_2),np.array(MOAC_3),np.array(MOAC_4)
				APOC_1, APOC_2, APOC_3, APOC_4 = np.array(APOC_1), np.array(APOC_2), np.array(APOC_3), np.array(APOC_4)
				STDOC_1, STDOC_2, STDOC_3, STDOC_4 = np.array(STDOC_1), np.array(STDOC_2), np.array(STDOC_3), np.array(STDOC_4)
				R_1, R_2, R_3 = np.array(R_1), np.array(R_2), np.array(R_3)
				# Various features are spliced into a line according to [feature 1 of channel 0, feature 1 of channel 1, feature 1 of channel 2, feature 1 of channel 3,
			    #                                                        feature 2 of channel 0, feature 2 of channel 1, feature 2 of channel 2, feature 2 of channel 3,...]
			    #                                                        array
				window_All_Feature = np.concatenate((iEMG,RMS,MAV,MAV1,MAV2,VAR,SSI,VOrder,DASDV,MAX,MIN,Range,LOG,WL,AAC,MFL,WAMP,ZC,SSC,Sk1,Sk2,Sk3,Sk4,Kr1,Kr2,Kr3,Kr4,
													 MPF,MF,PKF,MNP,TTP,
													 MOAC_1, MOAC_2, MOAC_3, MOAC_4,
													 APOC_1, APOC_2, APOC_3, APOC_4,
													 STDOC_1, STDOC_2, STDOC_3, STDOC_4,
													 R_1, R_2, R_3,
													 Beta,alpha1,alpha2,alpha3,alpha4))
				label = int(files[i][17]) # Get label
				window_All_Feature_label = np.append(window_All_Feature,label)  # Combine all characteristic values with corresponding labels
				window_All_Feature_label = window_All_Feature_label[np.newaxis,:] # Expand one-dimensional array to one dimension for easy splicing
				result = np.concatenate((result,window_All_Feature_label),axis=0)

		names = 'iEMG,RMS,MAV,MAV1,MAV2,VAR,SSI,VOrder,DASDV,MAX,MIN,Range,LOG,WL,AAC,MFL,WAMP,ZC,SSC,Sk1,Sk2,Sk3,Sk4,Kr1,Kr2,Kr3,Kr4,' \
				  'MPF,MF,PKF,MNP,TTP,' \
				  'MOAC_1,MOAC_2,MOAC_3,MOAC_4,APOC_1,APOC_2,APOC_3,APOC_4,STDOC_1,STDOC_2,STDOC_3,STDOC_4,R_1,R_2,R_3,' \
				  'Beta,alpha1,alpha2,alpha3,alpha4'
		columns = []
		for name in names.split(','):
			for channel in range(4):
				columns.append(name+'_ch'+str(channel))
		columns.append('label')
		result = pd.DataFrame(result[1:,:],columns=columns)  # Install data in pandas format
		result.to_csv(features_save_psth)
		end = time.time()
		print('Time consuming:', end - start)