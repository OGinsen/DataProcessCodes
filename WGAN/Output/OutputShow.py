import numpy as np
# import matplotlib
# matplotlib.rc('font',size = 35)
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import tensorflow as tf
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# Generated = np.loadtxt('G0/ch0/Generated_40300.csv',delimiter=',')
# Reference = np.loadtxt('G0/ch0/Reference_40300.csv',delimiter=',')
# noise = np.loadtxt('G0/ch0/Noise_40300.csv',delimiter=',')
Generated = np.loadtxt('GAN/G0/ch0/Generated_49999.csv',delimiter=',')
Reference = np.loadtxt('GAN/G0/ch0/Reference_49999.csv',delimiter=',')
noise = np.loadtxt('GAN/G0/ch0/Noise_49999.csv',delimiter=',')
# validated = tf.nn.sigmoid(
# 	tf.convert_to_tensor(np.loadtxt('Discriminated_49999.csv',delimiter=','))
# )

def fft_Spectrum_Map(y, N, Freq):
	"""
    N = 1024
    Freq = 512
    interval = 1.0 / Freq
    lenth = 2
    :return:
    1、FFT——离散傅里叶变换（DFT）的快速算法。它是根据离散傅氏变换的奇、偶、虚、实等特性，对离散傅立叶变换的算法进行改进获得的。
	2、假设采样频率为Fs，信号频率F，采样点数为N。那么FFT之后结果就是一个为N点的复数。每一个点就对应着一个频率点。这个点的模值，就是该频率值下的幅度特性。
	3、假设采样频率为Fs，采样点数为N，做FFT之后，某一点n（n从1开始）表示的频率为：Fn=(n-1)*Fs/N；该点的模值除以N/2就是对应该频率下的信号的幅度（对于直流信号是除以N）；
	该点的相位即是对应该频率下的信号的相位。相位的计算可用函数atan2(b,a)计算。atan2(b,a)是求坐标为(a,b)点的角度值，范围从-pi到pi。要精确到xHz，则需要采样长度为1/x秒的信号，并做FFT。
	要提高频率分辨率，就需要增加采样点数，这在一些实际的应用中是不现实的，需要在较短的时间内完成分析。解决这个问题的方法有频率细分法，比较简单的方法是采样比较短时间的信号，然后在后面补充一定数量的0，使其长度达到需要的点数，再做FFT，
	这在一定程度上能够提高频率分辨力。
	4、由于FFT结果的对称性，通常我们只使用前半部分的结果，即小于采样频率一半的结果。
    """
	interval = 1.0 / Freq
	lenth = N / Freq
	FFT = fft(y)    # 快速傅里叶变换得到一组复数
	FFT_real = FFT.real    # 获取实数部分
	FFT_imag = FFT.imag    # 获取虚数部分

	Original_amplitude = np.abs(FFT)  # 通过取模获取原始幅值
	amplitude = np.array(Original_amplitude/(N/2))    # 获取各频率所对应的幅值
	amplitude_1 = amplitude[:int(N/2)]      # 由于对称性，只取一半区间
	amplitude_1[0] /= 2

	frequency = np.arange(N)*Freq/N   # 计算频率
	frequency_1 = frequency[:int(N/2)]  # 单边谱的频率轴,只取一半区间
	return frequency_1 , amplitude_1

for i in range(Generated.shape[0]):
	print('i:',i)
	plt.figure(1,figsize=(20,15))
	plt.subplot(211)
	plt.plot(Generated[i],label='Generated signal')
	plt.ylabel('sEMG', fontsize=40)
	plt.xlabel('Sampling point', fontsize=40)
	plt.tick_params(labelsize=40)
	plt.legend(loc = 'lower right',fontsize=40,framealpha=0.4)
	plt.grid(linestyle='--')
	plt.subplot(212)
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.37)
	plt.plot(Reference[i],label='Reference signal')
	plt.ylabel('sEMG', fontsize=40)
	plt.xlabel('Sampling point', fontsize=40)
	plt.tick_params(labelsize=40)
	plt.legend(loc = 'lower right',fontsize=40,framealpha=0.4)
	plt.grid(linestyle='--')
	# plt.tight_layout()
	g_f, g_a = fft_Spectrum_Map(Generated[i][512:512 + 1024], len(Generated[i][512:512 + 1024]), 2000)
	r_f, r_a = fft_Spectrum_Map(Reference[i][512:512 + 1024], len(Reference[i][512:512 + 1024]), 2000)
	plt.figure(2,figsize=(20,15))
	plt.subplot(211)
	plt.stem(g_f, g_a, 'b-', 'C0.',label='Generated signal')
	plt.tick_params(labelsize=40)
	plt.ylabel('Magnitude', fontsize=40)
	plt.xlabel('Frequency (Hz)', fontsize=40)
	plt.legend(loc = 'upper right',fontsize=40,framealpha=0.4)
	plt.grid(linestyle='--')
	plt.subplot(212)
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=0.37)
	plt.stem(r_f, r_a, 'b-', 'C0.',label='Reference signal')
	plt.tick_params(labelsize=40)
	plt.ylabel('Magnitude', fontsize=40)
	plt.xlabel('Frequency (Hz)', fontsize=40)
	plt.legend(loc='upper right', fontsize=40, framealpha=0.4)
	plt.grid(linestyle='--')
	plt.show()
# plt.show()

# for i in range(noise.shape[0]):
# 	plt.figure(1)
# 	plt.plot(noise[i],label='Noise:F-sEMG')
# 	plt.ylabel('Normalized amplitude', fontsize=35)
# 	plt.xlabel('Sampling point', fontsize=35)
# 	plt.tick_params(labelsize=35)
# 	plt.legend(fontsize=35)
# 	plt.grid(linestyle='--')
#
# 	plt.figure(2)
# 	plt.plot(Reference[i],label='Reference signal:NF-sEMG')
# 	plt.ylabel('Normalized amplitude', fontsize=35)
# 	plt.xlabel('Sampling point', fontsize=35)
# 	plt.tick_params(labelsize=35)
# 	plt.legend(fontsize=35)
# 	plt.grid(linestyle='--')
#
# 	plt.figure(3)
# 	plt.plot(Generated[i],label='Generated signal')
# 	plt.ylabel('Normalized amplitude', fontsize=35)
# 	plt.xlabel('Sampling point', fontsize=35)
# 	plt.tick_params(labelsize=35)
# 	plt.legend(fontsize=35)
# 	plt.grid(linestyle='--')
#
# 	plt.show()
# g_loss = np.loadtxt('G6/ch0/g_loss_2.txt',delimiter=',')[:40000]
# d_loss = np.loadtxt('G6/ch0/d_loss_2.txt',delimiter=',')[:40000]
#
# FFT_MSE = np.loadtxt('G6/ch0/fft_metric_2.txt',delimiter=',')[:40000]
# DTW = np.loadtxt('G6/ch0/dtw_metric_2.txt',delimiter=',')[:40000]
#
# plt.figure(4)
# plt.title('Losses - Generator / Discriminator', fontsize=35)
# plt.plot(d_loss,label='Discriminator')
# plt.plot(g_loss,label='Generator')
# plt.xlabel('Epoch', fontsize=35)
# plt.ylabel('Loss/Metric', fontsize=35)
# plt.tick_params(labelsize=35)
# plt.legend(fontsize=35)
# plt.grid(linestyle='--')
#
# plt.figure(5)
# plt.title('FFT of Generated Signal', fontsize=35)
# plt.plot(FFT_MSE, label='MSE of  FFT')
# plt.xlabel('Epoch', fontsize=35)
# plt.ylabel('MSE of  FFT', fontsize=35)
# plt.tick_params(labelsize=35)
# plt.legend(fontsize=35)
# plt.grid(linestyle='--')
#
# plt.figure(6)
# plt.title('DTW Distance of Generated Signal', fontsize=35)
# plt.plot(DTW, label='DTW Distance')
# plt.xlabel('Epoch', fontsize=35)
# plt.ylabel('DTW Distance', fontsize=35)
# plt.tick_params(labelsize=35)
# plt.legend(fontsize=35)
# plt.grid(linestyle='--')
#
#
# plt.show()

