import numpy as np
import pywt


'''
Feature function
'''

##################################Time domain feature###################################################################
def Feature_iEMG(data):
	'''
	Integral EMG value
	:param data: Two dimensional array
	:return: One dimensional array with 4 numbers in a row
	'''
	iEMG = np.sum(abs(data), axis=0)
	return iEMG
def Feature_RMS(data):
	'''
	Root mean square
	:param data: Two dimensional array
	:return: One dimensional array with 4 numbers in a row
	'''
	RMS = np.sqrt(np.sum(np.square(data),axis=0)/len(data))
	return RMS
def Feature_MAV(data):
	'''
	Mean absolute value
	:param data: Two dimensional array
	:return: One dimensional array with 4 numbers in a row
	'''
	MAV = np.sum(abs(data), axis=0)/len(data)
	return MAV
def Feature_MAV1(data):
	'''
	Improved mean absolute value type 1
	:param data: Two dimensional array
	:return: One dimensional array with 4 numbers in a row
	'''
	Data = np.zeros(np.shape(data))
	for i in range(len(data)):
		if 0.25*len(data) <= i <= 0.75*len(data):
			Data[i,:] = 1 * abs(data[i,:])
		else:
			Data[i,:] = 0.5 * abs(data[i, :])
	MAV1 = np.sum(Data, axis=0) / len(Data)
	return MAV1
def Feature_MAV2(data):
	'''
	Improved mean absolute value type 2
	:param data: Two dimensional array
	:return: One dimensional array with 4 numbers in a row
	'''
	Data = np.zeros(np.shape(data))
	for i in range(len(data)):
		if 0.25*len(data) <= i <= 0.75*len(data):
			Data[i,:] = 1*abs(data[i,:])
		elif i < 0.25*len(data):
			Data[i, :] = ((4*i)/len(data)) * abs(data[i, :])
		else:
			Data[i, :] = 4*(i-len(data))/len(data) * abs(data[i, :])
	MAV2 = np.sum(Data, axis=0) / len(Data)
	return MAV2
def Feature_VAR(data):
	'''
	Variance
	:param data: Two dimensional array
	:return: One dimensional array with 4 numbers in a row
	'''
	VAR = np.sum(data**2,axis=0)/(len(data)-1)
	return VAR
def Feature_SSI(data):
	'''
	Simple square integral
	:param data: Two dimensional array
	:return: One dimensional array with 4 numbers in a row
	'''
	SSI = np.sum(data**2,axis=0)
	return SSI
def Feature_VOrder(data,v=3):
	'''
	V-order is a nonlinear characteristic index
	:param data: Two dimensional array
	:return: One dimensional array with 4 numbers in a row
	'''
	X = np.sum(data**v,axis=0)/len(data)
	for c in range(len(X)):
		if X[c] < 0:
			X[c] = -abs(X[c])**(1/v)
		else:
			X[c] = X[c] ** (1 / v)
	V = X
	# print('V:',V)
	return V
def Feature_DASDV(data):
	'''
	Absolute standard deviation of difference
	:param data: Two dimensional array
	:return: One dimensional array with 4 numbers in a row
	'''
	Data = np.zeros(np.shape(data))
	for i in range(len(data)-1):
		Data[i,:] = data[i+1,:] - data[i,:]
	DASDV = np.sqrt(np.sum(np.square(Data[:-1,:]),axis=0)/(len(Data)-1))
	return DASDV
def Feature_MAX(data):
	'''
	Maximum value
	:param data: Two dimensional array
	:return: One dimensional array with 4 numbers in a row
	'''
	MAX = np.max(data,axis=0)
	return MAX
def Feature_MIN(data):
	'''
	Minimum
	:param data: Two dimensional array
	:return: One dimensional array with 4 numbers in a row
	'''
	MIN = np.min(data,axis=0)
	return MIN
def Feature_range(data):
	'''
	Range
	:param data: Two dimensional array
	:return: One dimensional array with 4 numbers in a row
	'''
	R = np.max(data,axis=0) - np.min(data,axis=0)
	return R
def Feature_LOG(data):
	'''
	Logic detection
	:param data: Two dimensional array
	:return: One dimensional array with 4 numbers in a row
	'''
	LOG = np.exp(np.sum(np.log(abs(data)),axis=0)/len(data))
	return LOG
def Feature_WL(data):
	'''
	Waveform length
	:param data: Two dimensional array
	:return: One dimensional array with 4 numbers in a row
	'''
	Data = np.zeros(np.shape(data))
	for i in range(len(data)-1):
		Data[i,:] = abs(data[i+1,:] - data[i,:])
	WL = np.sum(Data[:-1,:],axis=0)
	return WL
def Feature_AAC(data):
	'''
	Average amplitude change
	:param data: Two dimensional array
	:return: One dimensional array with 4 numbers in a row
	'''
	Data = np.zeros(np.shape(data))
	for i in range(len(data)-1):
		Data[i,:] = abs(data[i+1,:] - data[i,:])
	AAC = np.sum(Data[:-1,:],axis=0)/(len(Data)-1)
	return AAC
def Feature_MFL(data):
	'''
	Maximum fractal length
	:param data: Two dimensional array
	:return: One dimensional array with 4 numbers in a row
	'''
	Data = np.zeros(np.shape(data))
	for i in range(len(data)-1):
		Data[i,:] = (data[i+1,:] - data[i,:])**2
	MFL = np.log10(np.sqrt(np.sum(Data[:-1,:],axis=0)))
	return MFL
def Feature_WAMP(data,th):
	'''
	Willison amplitude
	:param data: Two dimensional array
	:param th: Threshold
	:return: One dimensional array with 4 numbers in a row
	'''
	Data = np.zeros(np.shape(data))
	for i in range(len(data)-1):
		for ch in range(np.shape(data)[1]):
			if abs(data[i+1,ch] - data[i,ch])>th:
				Data[i,ch] = 1
			else:
				Data[i,ch] = 0
	WAMP = np.sum(Data[:-1,:],axis=0)
	return WAMP
def Feature_ZC(data,th):
	'''
	Zero-crossing rate
	:param data: Two dimensional array
	:param th: Threshold
	:return: One dimensional array with 4 numbers in a row
	'''
	Data = np.zeros(np.shape(data))
	for i in range(len(data)-1):
		for ch in range(np.shape(data)[1]):
			if  data[i+1,ch]*data[i,ch]<0 and abs(data[i+1,ch] - data[i,ch])>th:
				Data[i,ch] = 1
			else:
				Data[i, ch] = 0
	ZC = np.sum(Data[:-1,:],axis=0)
	return ZC
def Feature_SSC(data,th):
	'''
	Slope sign change
	:param data: Two dimensional array
	:param th: Threshold
	:return: One dimensional array with 4 numbers in a row
	'''
	Data = np.zeros(np.shape(data))
	for i in range(1,len(data)-1):
		for ch in range(np.shape(data)[1]):
			if (data[i,ch]-data[i+1,ch])*(data[i,ch]-data[i-1,ch])>= th:
				Data[i,ch] = 1
			else:
				Data[i,ch] = 0
	SSC = np.sum(Data[1:-1,:],axis=0)
	return SSC
def Feature_SK(data):
	'''
	Skewness
	:param data: Two dimensional array
	:return: Four kinds of skewness, four one-dimensional arrays
	'''
	Sk1 = len(data)/((len(data)-1)*(len(data)-2)*np.std(data,axis=0)**3)*np.sum((data-np.mean(data,axis=0))**3)
	# Sk1 = (np.sum((data - np.mean(data, axis=0)) ** 3, axis=0) / len(data)) / (np.sum((data - np.mean(data, axis=0)) ** 2, axis=0) / len(data)) ** 1.5  #  如果用stats.skew()计算就等同于这行计算
	q1, q2, q3 = np.percentile(data,(25,50,75),axis=0)
	Sk2 = ((q3-q2) - (q2-q1))/(q3-q1)
	Sk3 = (np.mean(data,axis=0)-q2)/np.mean(abs(data - q2), axis=0)
	Sk4 = (np.mean(data,axis=0)-q2)/np.var(data,axis=0)
	return Sk1,Sk2,Sk3,Sk4
def Feature_Kr(data):
	'''
	Kurtosis
	:param data: Two dimensional array
	:return: Four kinds of kurtosis, four one-dimensional arrays
	'''
	N = len(data)
	Kr1 = (N*(N+1)/((N-1)*(N-2)*(N-3)*np.std(data,axis=0)**4))*np.sum((data-np.mean(data,axis=0)**4),axis=0)-(3*(N-1)**2)/((N-2)*(N-3))
	q1,q2,q3,q4,q5,q6= np.percentile(data, (12.5,25.0,37.5,62.5,75.0,87.5), axis=0)
	Kr2 = ((q6-q4)+(q3-q1))/(q5-q2)
	e1,e2,e3,e4,e5,e6,e7 = np.percentile(data,(20,30,35,50,65,70,80),axis=0)
	Kr3 = []
	for ch in range(np.shape(data)[1]):
		Kr3.append((np.mean(data[:,ch][data[:,ch]>e5[ch]])-np.mean(data[:,ch][data[:,ch]<e3[ch]]))/(
			np.mean(data[:,ch][data[:,ch]>e4[ch]]) - np.mean(data[:,ch][data[:,ch]<e4[ch]])
		))
	Kr3 = np.array(Kr3)
	Kr4 = (e7-e1)/(e6-e2)
	return Kr1,Kr2,Kr3,Kr4

##################################Frequency domain feature##############################################################
def Feature_MPF(f,p):
	'''
	Mean power frequency
	:param f: One dimensional array  ,  Frequency
	:param p: One dimensional array  ,  Power
	:return: Mean power frequency
	'''
	MPF = np.sum(f*p)/np.sum(p)
	return MPF
def Feature_MF(f,p):
	'''
	Median frequency
	:param f: One dimensional array  ,  Frequency
	:param p: One dimensional array  ,  Power
	:return: Median frequency
	'''
	x = []
	for i,j in enumerate(p):
		x.append(abs(np.sum(p[:i+1])-np.sum(p)/2))
	for i, j in enumerate(p):
		if abs(np.sum(p[:i+1])-np.sum(p)/2)  == np.min(x):
			# print(np.sum(p[:i+1]),np.sum(p[i:]),np.sum(p)/2,abs(np.sum(p[:i+1])-np.sum(p)/2))
			# print('MDF:',f[i])
			return f[i]
def Feature_PKF(p):
	'''
	Peak frequency
	:param p: One dimensional array  ,  Power
	:return:
	'''
	PKF = np.max(p)
	return PKF
def Feature_MNP(p):
	'''
	Mean power
	:param p: One dimensional array  ,  Power
	:return:
	'''
	MNP = np.sum(p)/len(p)
	return MNP
def Feature_TTP(p):
	'''
	Total power
	:param p: One dimensional array  ,  Power
	:return:
	'''
	TTP = np.sum(p)
	return TTP

##################################Time-frequency feature################################################################
def Wavelet_transform(data,wavelet='db4',level=3):
	'''
	Wavelet transform
	:param data: One-dimensional array signal
	:param wavelet: Wavelet basis function
	:param level: Decomposition layers
	:return: Wavelet coefficients of wavelet subbands
	'''
	coeffs = pywt.wavedec(data,wavelet,level) # Wavelet coefficients of wavelet subbands are obtained by multi-level wavelet decomposition
	# print('len(coeffs):',len(coeffs))
	# cA3, cD3, cD2, cD1 = coeffs # If 3 layers are decomposed, the high frequency coefficients (cA3, cD3, cD2) of 3 subbands and the low frequency coefficients (cD1) of one subband are obtained.
	# y = pywt.waverec(coeffs, 'db4')  # Inverse transform reconstructed signal
	return coeffs
def Feature_MOAC(data):
	'''
	Absolute mean of eigenvalue coefficients
	:param data: One-dimensional array signal
	:return: 4 characteristic values
	'''
	cA3, cD3, cD2, cD1 = pywt.wavedec(data,wavelet='db4',level=3)
	# cA3, cD3, cD2, cD1 = Wavelet_transform(data,wavelet='db4',level=3)
	MOAC_1 = np.sum(np.abs(cD1)) / len(cD1)
	MOAC_2 = np.sum(np.abs(cD2)) / len(cD2)
	MOAC_3 = np.sum(np.abs(cD3)) / len(cD3)
	MOAC_4 = np.sum(np.abs(cA3)) / len(cA3)
	return MOAC_1,MOAC_2,MOAC_3,MOAC_4
def Feature_APOC(data):
	'''
	Average energy of eigenvalue coefficients
	:param data: One-dimensional array signal
	:return: 4 characteristic values
	'''
	# cA3, cD3, cD2, cD1 = Wavelet_transform(data, wavelet='db4', level=3)
	cA3, cD3, cD2, cD1 = pywt.wavedec(data, wavelet='db4', level=3)
	APOC_1 = np.sum(cD1 ** 2) / len(cD1)
	APOC_2 = np.sum(cD2 ** 2) / len(cD2)
	APOC_3 = np.sum(cD3 ** 2) / len(cD3)
	APOC_4 = np.sum(cA3 ** 2) / len(cA3)
	return APOC_1,APOC_2,APOC_3,APOC_4
def Feature_STDOC(data):
	'''
	Standard deviation of eigenvalue coefficients
	:param data: One-dimensional array signal
	:return: 4 characteristic values
	'''
	# cA3, cD3, cD2, cD1 = Wavelet_transform(data, wavelet='db4', level=3)
	cA3, cD3, cD2, cD1 = pywt.wavedec(data, wavelet='db4', level=3)
	STDOC_1 = np.sqrt(np.sum((cD1 - np.mean(cD1))**2) / len(cD1))
	STDOC_2 = np.sqrt(np.sum((cD2 - np.mean(cD2))**2) / len(cD2))
	STDOC_3 = np.sqrt(np.sum((cD3 - np.mean(cD3))**2) / len(cD3))
	STDOC_4 = np.sqrt(np.sum((cA3 - np.mean(cA3))**2) / len(cA3))
	return STDOC_1,STDOC_2,STDOC_3,STDOC_4
def Feature_R(data):
	'''
	Ratio of mean values of eigenvalue coefficients
	:param data: One-dimensional array signal
	:return: 3 characteristic values
	'''
	MOAC_1,MOAC_2,MOAC_3,MOAC_4 = Feature_MOAC(data)
	R_1 = MOAC_2/MOAC_1
	R_2 = MOAC_3/MOAC_2
	R_3 = MOAC_4/MOAC_3
	return R_1,R_2,R_3

########################Feature extraction of sEMG raw signal based on parameter model##################################
from statsmodels.tsa.ar_model import AutoReg   #Autoregressive model
def Feature_AR(data):
	'''
	Fourth-order autoregressive model Coefficients α_i (i = 1, 2, 3, 4) and β as features
	:param data: Two dimensional array
	:return: A two-dimensional array with five rows (five eigenvalues) and four columns (four channels),
	'''
	Features = []
	for ch in range(np.shape(data)[1]):
		model = AutoReg(data[:,ch], lags=[1, 2, 3, 4])
		model_fit = model.fit()
		[Beta, alpha1, alpha2, alpha3, alpha4] = model_fit.params
		Features.append(np.array([Beta, alpha1, alpha2, alpha3, alpha4]))
	return np.array(Features).T
