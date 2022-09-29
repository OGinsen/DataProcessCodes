from scipy import signal
from scipy.signal.signaltools import filtfilt
import numpy as np
from scipy.fftpack import fft

'''
Butterworth filter and Normalized least mean square adaptive filter function
'''

def Butterworth_Filter(Raw_1):
	b_s_1, a_s_1 = signal.butter(4, [0.058, 0.060], btype='bandstop')    #  Filter 59HZ 0.058 = 58/(2000/2)  0.060 = 60/(2000/2)
	b_s_2,a_s_2 = signal.butter(4, [0.049, 0.051], btype='bandstop') #  Filter 50HZ
	b,a = signal.butter(8,[0.020,0.500],'bandpass')  # Passband 20-500HZ

	raw_1 = filtfilt(b_s_1, a_s_1, Raw_1)
	raw_1  = filtfilt(b_s_2,a_s_2,raw_1)
	raw_1 = filtfilt(b,a,raw_1)
	return raw_1

def NLMS(xn, dn, M , mu, itr):
	"""
    Using Normal LMS adaptive filtering
    :param xn: The input signal sequence is the resting signal
    :param dn: The desired response sequence is the measured active segment signal with noise
    :param M: Order of filter
    :param mu: Convergence factor (step size)
    :param itr: Number of iterations
    :return:
    """
	en = np.zeros(itr)  # Error sequence, en (k) represents the error between the expected output and the actual input at the k-th iteration
	W = np.zeros((M, itr))  # Each row represents a weighting parameter, and each column represents - iterations, with an initial value of 0
	# Iterative calculation
	for k in range(M, itr):
		x = xn[k:k - M:-1]
		y = np.matmul(W[:, k - 1], x)
		en[k] = dn[k] - y
		W[:, k] = W[:, k - 1] + 2 * mu * en[k] * x / (np.sum(np.multiply(x, x)) + 1e-10)
	# Find the optimal output sequence
	yn = np.inf * np.ones(len(xn))
	for k in range(M, len(xn)):
		x = xn[k:k - M:-1]
		yn[k] = np.matmul(W[:, -1], x)
	return yn, W, en

def LMS(xn, dn, M, mu, itr):
	"""
	Using LMS adaptive filtering
	:param xn: Input signal sequence
	:param dn: Expected response sequence
	:param M: Order of filter
	:param mu: Convergence factor (step size)
	:param itr: Number of iterations
	:return:
	"""
	en = np.zeros(itr)  # Error sequence, en (k) represents the error between the expected output and the actual input at the k-th iteration
	W = np.zeros((M, itr))  # Each row represents a weighting parameter, and each column represents - iterations, with an initial value of 0
	# Iterative calculation
	for k in range(M, itr):
		x = xn[k:k - M:-1]
		y = np.matmul(W[:, k - 1], x)
		en[k] = dn[k] - y
		W[:, k] = W[:, k - 1] + 2 * mu * en[k] * x
	# Find the optimal output sequence
	yn = np.inf * np.ones(len(xn))
	for k in range(M, len(xn)):
		x = xn[k:k - M:-1]
		yn[k] = np.matmul(W[:, -1], x)
	return yn, W, en


def fft_Spectrum_Map(y, N, Freq):
	'''
	Fourier Transform Spectrogram
	:param y: One-dimensional signal
	:param N: Sampling points
	:param Freq: Sampling frequency
	:return: frequency, amplitude
	'''
	interval = 1.0 / Freq
	lenth = N / Freq
	FFT = fft(y)    # Fast Fourier Transform to get a set of complex numbers
	FFT_real = FFT.real    # get real part
	FFT_imag = FFT.imag    # get the imaginary part

	Original_amplitude = np.abs(FFT)  # Get the original amplitude by taking the modulo
	amplitude = np.array(Original_amplitude/(N/2))    # Get the amplitude corresponding to each frequency
	amplitude_1 = amplitude[:int(N/2)]      # Due to symmetry, only half of the interval is taken
	amplitude_1[0] /= 2

	frequency = np.arange(N)*Freq/N   # Calculate frequency
	frequency_1 = frequency[:int(N/2)]  # The frequency axis of the single-sided spectrum, only half of the interval is taken
	return frequency_1 , amplitude_1

def Power_spectrum(data,N,Freq):
	'''
	Calculate power spectrum
	:param data: One-dimensional signal
	:param N: Sampling points
	:param Freq: Sampling frequency
	:return: frequency, power
	'''
	frequency , amplitude = fft_Spectrum_Map(data,N,Freq)
	power = amplitude**2/N
	return frequency , power