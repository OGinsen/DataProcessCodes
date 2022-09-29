import numpy as np
import argparse
import json
from WGAN.Models.WGAN_Discriminator import Discriminator
from WGAN.Models.WGAN_Generator import Generator
import tensorflow as tf
from Classification.DeepLearning.CNN_Models import ResNet18
from ActiveSegmentDetection.ShortEnergyFun import Active_Segment_Detection
from Denoise.DenoiseFun import NLMS,fft_Spectrum_Map
from scipy import signal
from scipy.signal.signaltools import filtfilt
import os
import random

import matplotlib.pyplot as plt
def Butterworth_Filter(Raw_1):
	b,a = signal.butter(8,[0.020,0.500],'bandpass')  #  #带通滤波器，通带5-200HZ,0.01=5/(sampling_rate/2),0.4=200/(sampling_rate/2)，3阶带通
	raw_1 = filtfilt(b,a,Raw_1)
	return raw_1
def Save_and_Filtering_Active_segment(list_x,list_y,data,rest):
	#Firstly, the active segment is intercepted, and then the adaptive filtering NLMS is used
	start_end = []
	Coordinate_set = list(zip(list_x, list_y))
	for i in range(len(Coordinate_set)):
		if Coordinate_set[i][1] > 0:
			start_end.append(Coordinate_set[i][0])
	for i in range(0, len(start_end), 2):
		Active_segment = data[start_end[i]:start_end[i + 1], :]
		A_f, A_a = fft_Spectrum_Map(Active_segment[:,0], len(Active_segment[:,0]), 2000)
		plt.figure(2)
		plt.subplot(411)
		plt.stem(A_f, A_a, 'b-', 'C0.')
		plt.grid()
		plt.subplot(413)
		plt.grid()
		# plt.figure(1)
		plt.plot(Active_segment[:, 0])
		# plt.show()
		for c in range(Active_segment.shape[1]):
			[yn,Wn,en] = NLMS(rest[:len(Active_segment[:,c]),c],Active_segment[:,c],64,0.03,len(Active_segment[:,c]))
			Active_segment[:,c] = en
			Active_segment[:, c] = Butterworth_Filter(Active_segment[:, c]) ##The components below 20Hz and above 500Hz are filtered by band-pass filter
			filtered_f, filtered_a = fft_Spectrum_Map(Active_segment[:,0], len(Active_segment[:,0]), 2000)
			plt.figure(2)
			plt.subplot(412)
			plt.stem(filtered_f, filtered_a, 'b-', 'C0.')
			plt.grid()
			plt.subplot(413)
			plt.grid()
			# plt.figure(1)
			plt.plot(Active_segment[:, c])
			# plt.show()
		plt.savefig('111.png')
def EvaluateModels(config,start_epoch,end_epoch,ch,G):
	generator = Generator(config,training = False)
	# discriminator = Discriminator(config, training=False)

	model_save_path = '../Classification/DeepLearning/SavedModels/relax_before/ch0/checkpoint_ResNet18/ResNet18_ch0.ckpt'

	model = ResNet18([2, 2, 2, 2])
	model.load_weights(model_save_path)

	All_accuracy = []
	for index in range(start_epoch,end_epoch+50,50):
		generator.load(index)
		# discriminator.load(index)

		# tired_emg = np.load('./ActiveDatasetsPadded/tired/S1/S1_t_ActiveDatasetsPadded_AllG.npy')[:,:,ch:ch+1]
		# relax_emg = np.load('./ActiveDatasetsPadded/relax/S1/S1_r_ActiveDatasetsPadded_AllG.npy')[:,:,ch:ch+1]
		# tired_emg_label = np.load('./ActiveDatasetsPadded/tired/S1/S1_t_ActiveDatasetsPadded_AllG_Label.npy')
		# relax_emg_label = np.load('./ActiveDatasetsPadded/relax/S1/S1_r_ActiveDatasetsPadded_AllG_Label.npy')

		# tired_emg = np.load('./ActiveDatasetsWindowsPaddedValidationSet/tired/AllS_t_ActiveDatasetsWindowsPaddedValidationSet_G0.npy')[:,:,ch:ch+1]
		# relax_emg = np.load('./ActiveDatasetsWindowsPaddedValidationSet/relax/S1/S1_r_ActiveDatasetsPadded_AllG.npy')[:,:,ch:ch+1]
		# tired_emg_label = np.load('./ActiveDatasetsWindowsPaddedValidationSet/tired/AllS_t_ActiveDatasetsWindowsPaddedValidationSet_G0_Label.npy')
		# relax_emg_label = np.load('./ActiveDatasetsWindowsPaddedValidationSet/relax/S1/S1_r_ActiveDatasetsPadded_AllG_Label.npy')
		tired_emg = np.load('ActiveDatasetsWindowsPadded/relax_before/All_tired_windows_0.npy')[:,:,ch:ch+1]
		print('tired_emg.shape:',tired_emg.shape)

		tf.random.set_seed(22)
		np.random.seed(22)
		idx = np.random.randint(0, tired_emg.shape[0], config['batch_size'])

		# reference_emg = relax_emg[idx]
		# labels = tired_emg_label[idx]
		# print('labels.shape:',labels.shape)

		noise = tired_emg[idx]
		generate_emg = generator.predict(noise)
		print('generate_emg.shape:', generate_emg.shape)

		# validated = discriminator.model(generate_emg)
		# validated = tf.nn.sigmoid(validated)
		# print('validated:', validated)


		generate_emg = np.reshape(generate_emg, (generate_emg.shape[0], generate_emg.shape[1]))
		# reference_emg = np.reshape(reference_emg, (reference_emg.shape[0], reference_emg.shape[1]))
		noise = np.reshape(noise, (noise.shape[0], noise.shape[1]))

		# labels = np.reshape(labels, (labels.shape[0], 1))
#########################################################################################################################
		# E, x, y, raw_x, raw_y = Active_Segment_Detection(generate_emg, 64, 32,
		# 												 th1=6e-06,
		# 												 th2=3e-06,
		# 												 channel=0)
		# A
########################################################################################################################
		true = []
		false = []
		for i in range(len(generate_emg)):
			emg_predict = generate_emg[i, 512:512 + 1024]
			print('emg_predict.shape:', emg_predict.shape)
			emg_predict = emg_predict.reshape(1, emg_predict.shape[0], 1)
			print('emg_predict.shape:',emg_predict.shape)
			result = model.predict(emg_predict)
			pred = tf.argmax(result, axis=1)
			print('pred:', pred)
			np_pred = np.array(pred)

			# label = labels[i]
			label = G
			if np_pred[0] == label:
				true.append(i)
			else:
				false.append(i)
			print('false:', false)
		Accuracy = len(true) / len(generate_emg)
		print('Accuracy:', Accuracy)
		print('len(pathDir):', len(generate_emg))
		print('len(true):', len(true))
		All_accuracy.append(Accuracy)
	print('All_accuracy:', All_accuracy)
	print('Maximum accuracy:%f, epoch:%f' % (np.max(All_accuracy), np.argmax(All_accuracy) * 50 + start_epoch))
	for idx,acc in enumerate(All_accuracy):
		if acc >= 0.70:
			print('epoch:',idx*50+start_epoch,' acc:',acc)
def TestModels(config,index,G,ch):
	generator = Generator(config,training = False)
	# discriminator = Discriminator(config, training=False)
	generator.load(index,G=G,ch=ch)
	padding_path = r'D:\PycharmProjects\AI\A_Experimental_data_processing\Datasets_Norm\aojx\relax\norm_relax_0.txt'
	padding = np.loadtxt(padding_path, delimiter=',', skiprows=0)[1:20000,:]

	model_save_path = '../Classification/DeepLearning/SavedModels/relax_before/ch' + str(ch) + '/checkpoint_ResNet18/ResNet18_ch'+ str(ch) +'.ckpt'

	model = ResNet18([2, 2, 2, 2])
	model.load_weights(model_save_path)

	TestDatas_path = 'D:\All_Datasets\CNNValidationSet\\tired\\'
	pathDir = os.listdir(TestDatas_path)
	true = []
	false = []
	num = 0
	for i in pathDir:
		if i[-3:] == 'csv' and i[11:11+7] == 'tired_' + str(G):
			num += 1
			emg_path = TestDatas_path + str(i)
			noise = np.loadtxt(emg_path, delimiter=',')[:,ch:ch+1]
			noise = np.concatenate((padding[:512,ch:ch+1], noise, padding[-512:,ch:ch+1]), axis=0)

			noise = noise.reshape(1, noise.shape[0], noise.shape[1])

			generate_emg = generator.predict(noise)[:,512:512+1024,:]

			print('generate_emg.shape:',generate_emg.shape)

			result = model.predict(generate_emg)
			pred = tf.argmax(result, axis=1)
			np_pred = np.array(pred)

			label = G
			if np_pred[0] == label:
				true.append(i)
			else:
				false.append(i)
			print('false:', false)
			print('label:', label, '===> pred:', np_pred[0])
	Accuracy = len(true) / num
	print('Accuracy:', Accuracy)
	print('num:', num)
	print('len(true):', len(true))

def Generate_and_Concate(config,index:list,G):
	generator_ch0 = Generator(config,training = False)
	# discriminator_ch0 = Discriminator(config, training=False)
	generator_ch1 = Generator(config,training = False)
	# discriminator_ch1 = Discriminator(config, training=False)
	generator_ch2 = Generator(config,training = False)
	# discriminator_ch2 = Discriminator(config, training=False)
	generator_ch3 = Generator(config,training = False)
	# discriminator_ch3 = Discriminator(config, training=False)

	generator_ch0.load(index=index[0],G=G,ch=0)
	# discriminator_ch0.load(index[0])
	generator_ch1.load(index=index[1],G=G,ch=1)
	# discriminator_ch1.load(index[1])
	generator_ch2.load(index=index[2],G=G,ch=2)
	# discriminator_ch2.load(index[2])
	generator_ch3.load(index=index[3],G=G,ch=3)
	# discriminator_ch3.load(index[3])

	model_save_path = r'../Classification/DeepLearning/SavedModels/relax_before/4ch/checkpoint_ResNet18/ResNet18_4ch.ckpt'
	model = ResNet18([2, 2, 2, 2])
	model.load_weights(model_save_path)

	TestDatas_path = 'D:\All_Datasets\CNNValidationSet\\tired\\'
	# TestDatas_path = 'D:\All_Datasets\CNNValidationSet\\tired_before\\'
	pathDir = os.listdir(TestDatas_path)
	true = []
	false = []
	num = 0
	padding_path = r'D:\PycharmProjects\AI\A_Experimental_data_processing\Datasets_Norm\aojx\relax\norm_relax_0.txt'
	padding = np.loadtxt(padding_path, delimiter=',', skiprows=0)[1:20000,:]
	for i in pathDir:
		if i[-3:] == 'csv' and i[11:11+7] == 'tired_' + str(G):
		# if i[-3:] == 'csv' and i[:7] == 'tired_' + str(G):
			num += 1
			emg_path = TestDatas_path + str(i)
			noise = np.loadtxt(emg_path, delimiter=',')

			noise = np.concatenate((padding[:512,:], noise, padding[-512:,:]), axis=0)
			noise = noise.reshape(1, noise.shape[0], noise.shape[1])

			gen_ch0 = generator_ch0.predict(noise[:, :, 0:1])
			gen_ch1 = generator_ch1.predict(noise[:, :, 1:2])
			gen_ch2 = generator_ch2.predict(noise[:, :, 2:3])
			gen_ch3 = generator_ch3.predict(noise[:, :, 3:4])
			gen_4ch = np.concatenate((gen_ch0, gen_ch1, gen_ch2, gen_ch3), axis=2)
			# print('gen_4ch.shape:', gen_4ch.shape)

			result = model.predict(gen_4ch[:,512:512+1024,:])
			pred = tf.argmax(result, axis=1)
			np_pred = np.array(pred)
			label = G
			if np_pred[0] == label:
				true.append(i)
			else:
				false.append(i)
			if len(false) >= 231 :
				return -1
			print('false:', false)

	Accuracy = len(true) / num
	print('Accuracy:', Accuracy)
	print('num:', num)
	print('len(true):', len(true))
	return Accuracy

def GenerateAndSave(config,epoch,ch,numbers,G):
	generator = Generator(config, training=False)
	# discriminator = Discriminator(config, training=False)

	# generator.load(epoch)
	# discriminator.load(epoch)
	generator.load(epoch)

	# tired_emg = np.load('./ActiveDatasetsPadded/tired/S1/S1_t_ActiveDatasetsPadded_AllG.npy')[:, :,
	# 			ch:ch + 1]
	# relax_emg = np.load('./ActiveDatasetsPadded/relax/S1/S1_r_ActiveDatasetsPadded_AllG.npy')[:, :,
	# 			ch:ch + 1]
	# # tired_emg_label = np.load('./ActiveDatasetsWindowsPadded/tired/S1/S1_t_ActiveDatasetsWindowsPadded_AllG_Label.npy')
	# relax_emg_label = np.load('./ActiveDatasetsPadded/relax/S1/S1_r_ActiveDatasetsPadded_AllG_Label.npy')
	tired_emg = np.load('ActiveDatasetsWindowsPadded/tired_before/All_tired_windows_'+str(G)+'.npy')[:, :, ch:ch + 1]
	print('tired_emg.shape:', tired_emg.shape)

	tf.random.set_seed(22)
	np.random.seed(22)
	for num in range(numbers):
		idx = np.random.randint(0, tired_emg.shape[0], 1)
		# reference_emg = relax_emg[idx]
		# label = relax_emg_label[idx]
		label = G
		noise = tired_emg[idx]
		gen_emg = generator.model(noise)

		# validated = discriminator.model(gen_emg)
		# validated = tf.nn.sigmoid(validated)
		# print('validated:',validated)

		gen_emg = np.reshape(gen_emg, (gen_emg.shape[0], gen_emg.shape[1]))
		# label = np.reshape(label, (label.shape[0], 1))
		# gen_emg = np.concatenate((gen_emg, label), axis=1)

		np.savetxt('GeneratedSignals/' + 'Generated_emg_' + str(label) + '_' + str(num) + '.csv', gen_emg, delimiter=",")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='EMG-GAN - Generate EMG signals based on pre-trained model')

	parser.add_argument('--config_json', '-config', default='configuration.json', type=str,
						help='configuration json file path')

	args = parser.parse_args()

	config_file = args.config_json
	with open(config_file) as json_file:
		config = json.load(json_file)

	# EvaluateModels(config,0,200,0,0)
	# TestModels(config,index=200,G=1,ch=0)
	# Generate_and_Concate(config,index=[40300,2250,25850,31700],G=0) # 0.9376623376623376
	# Generate_and_Concate(config, index=[8450, 17300, 29750, 19200], G=1) # 0.8597402597402597
	# Generate_and_Concate(config, index=[26550, 49800, -1, -1], G=2) # 0.9662337662337662
	# Generate_and_Concate(config, index=[41050, 12450, 49650, 44400], G=3) # 0.8519480519480519
	# Generate_and_Concate(config, index=[49950, 20100, 47200, -1], G=4) # 0.9428571428571428
	# Generate_and_Concate(config, index=[40000, 49950, 49700, 49950], G=5)
	# Generate_and_Concate(config, index=[29800, 39800, 21650, 48650], G=5) # 0.7480519480519481

	ch0 = [24750,29450,29500,29550,29600,29700,29750,29800,29850,29900,29950,30000,30100,31100,38200,39150,39600,39700,40000,
		   41500,42000,42200,42550,42700,44650,45000,45050,47400,49400,49700,49800,49850,49950]
	ch1 = [12200,17850,23600,24850,25300,30000,31000,33300,33900,34750,34850,34900,34950,35000,35100,36950,39100,36950,39100,
		   39800,39900,41000,41100,41200,42400,43900,44200,44700,44750,44800,45000,45100,47850,48400,48850,49150,49800,49950]
	ch2 = [21650,22950,25300,38400,44650,47600,48750,49650,49700,49850,49950]
	ch3 = [29950,43450,46550,48650,48700,49100,49950]
	acc = 0
	Acc = []
	idx = []
	for ch0_idx in range(len(ch0)):
		for ch1_idx in range(len(ch1)):
			for ch2_idx in range(len(ch2)):
				for ch3_idx in range(len(ch3)):
					index = [ch0[ch0_idx],ch1[ch1_idx],ch2[ch2_idx],ch3[ch3_idx]]
					if index == [ch0[0],ch1[0],ch2[0],ch3[0]] or index == [ch0[0],ch1[0],ch2[0],ch3[1]] or index == [ch0[0],ch1[0],ch2[0],ch3[2]] or index == [ch0[0],ch1[0],ch2[0],ch3[3]] or index == [ch0[0],ch1[0],ch2[0],ch3[4]] or index == [ch0[0],ch1[0],ch2[0],ch3[5]] or index == [ch0[0],ch1[0],ch2[0],ch3[6]]:
						continue
					print('index:', index)
					acc = Generate_and_Concate(config, index=index, G=5)
					print('acc:',acc)
					if acc > 0.6:
						Acc.append(acc)
						idx.append(index)
						print('max_Acc:', max(Acc))
						print('max_idx:', idx[Acc.index(max(Acc))])
					print('Acc:', Acc)
					print('idx:', idx)


	# while acc < 0.85:
	# 	index = [random.choice(ch0), random.choice(ch1), random.choice(ch2), random.choice(ch3)]
	# 	acc = Generate_and_Concate(config, index=index, G=5)
	# 	print('index:', index)
	# 	if acc > 0.6:
	# 		Acc.append(acc)
	# 		idx.append(index)
	# 	print('Acc:', Acc)
	# 	print('idx:',idx)
	#
	# print('index_last:',index)
	# print('acc:',acc)
	# GenerateAndSave(config,epoch=-1,ch=0,numbers=8355,G=6)