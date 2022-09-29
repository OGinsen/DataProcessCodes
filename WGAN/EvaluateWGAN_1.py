import numpy as np
import argparse
import json
from WGAN.Models.WGAN_Generator import Generator
import tensorflow as tf
from Classification.DeepLearning.CNN_Models import ResNet18

import os

def EvaluateModels(config,start_epoch,end_epoch,G,ch,batch_size):
	generator = Generator(config,training = False)
	# discriminator = Discriminator(config, training=False)

	model_save_path = '../Classification/DeepLearning/SavedModels/relax/ch'+str(ch)+'/checkpoint_ResNet18/ResNet18_ch'+str(ch)+'.ckpt'

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
		tired_emg = np.load('ActiveDatasetsWindowsPadded/tired/All_tired_windows_'+str(G)+'.npy')[:, :, ch:ch+1]
		print('tired_emg.shape:',tired_emg.shape)

		tf.random.set_seed(22)
		np.random.seed(22)
		idx = np.random.randint(0, tired_emg.shape[0], batch_size)

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
		# noise = np.reshape(noise, (noise.shape[0], noise.shape[1]))

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
				print('label:', label, '===> pred:', np_pred)
			print('false:', false)
		Accuracy = len(true) / len(generate_emg)
		print('Accuracy:', Accuracy)
		print('len(pathDir):', len(generate_emg))
		print('len(true):', len(true))
		All_accuracy.append(Accuracy)

	useful_acc = []
	Epoch = []
	for idx, acc in enumerate(All_accuracy):
		if acc >= 0.10:
			print('acc >= 0.10 epoch:', idx * 50 + start_epoch, ' acc:', acc)
			useful_acc.append(acc)
			Epoch.append(idx * 50 + start_epoch)
	print('Maximum accuracy:%f, epoch:%f' % (max(useful_acc), Epoch[useful_acc.index(max(useful_acc))]))
	print('Minimum accuracy:%f, epoch:%f' % (min(useful_acc), Epoch[useful_acc.index(min(useful_acc))]))
	print('All_accuracy:', All_accuracy)
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

def Generate_and_Concate_and_save(config,index:list,G):
	generator_ch0 = Generator(config,training = False)
	generator_ch1 = Generator(config,training = False)
	generator_ch2 = Generator(config,training = False)
	generator_ch3 = Generator(config,training = False)

	generator_ch0.load(index=index[0],G=G,ch=0)
	generator_ch1.load(index=index[1],G=G,ch=1)
	generator_ch2.load(index=index[2],G=G,ch=2)
	generator_ch3.load(index=index[3],G=G,ch=3)

	model_save_path = r'../Classification/DeepLearning/SavedModels/relax/4ch/checkpoint_ResNet18/ResNet18_4ch.ckpt'
	model = ResNet18([2, 2, 2, 2])
	model.load_weights(model_save_path)

	TestDatas_path = 'D:\All_Datasets\CNNValidationSet\\tired\\'
	# TestDatas_path = 'D:\All_Datasets\CNNValidationSet\\tired_before\\'
	pathDir = os.listdir(TestDatas_path)
	true = []
	false = []
	num = 0
	padding_path = r'D:\All_Datasets\Padding_Signal\norm_relax_0.txt'
	padding = np.loadtxt(padding_path, delimiter=',', skiprows=0)[1:20000,:]
	Gen_save_path = 'D:\All_Datasets\EnhancedValidationSet\\'
	for i in pathDir:
		if i[-3:] == 'csv' and i[11:11+7] == 'tired_' + str(G):
		# if i[-3:] == 'csv' and i[:7] == 'tired_' + str(G):
			num += 1
			emg_path = TestDatas_path + i
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
				print('pred:', np_pred[0])
			# if len(false) >= 231 :
			# 	return -1.
			print('len(false):', len(false) ,' num:', num,' Accuracy:', len(true) / num)

			gen_4ch = gen_4ch.reshape(gen_4ch.shape[1], gen_4ch.shape[2])
			np.savetxt(Gen_save_path + i[:-4] + '_Enhanced.csv', gen_4ch[512:512+1024,:], delimiter=',')

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

	# EvaluateModels(config,45000,49950, 5, 3, 128)
	Generate_and_Concate_and_save(config,index=[40300,2250,25850,31700],G=0) # 0.9246753246753247
	Generate_and_Concate_and_save(config, index=[8450, 17300, 29750, 19200], G=1) # 0.9636363636363636
	Generate_and_Concate_and_save(config, index=[26550, 49800, -1, -1], G=2) # 0.9753246753246754
	Generate_and_Concate_and_save(config, index=[41050, 12450, 49650, 44400], G=3) # 0.9298701298701298
	Generate_and_Concate_and_save(config, index=[49950, 20100, 47200, -1], G=4) # 0.9272727272727272
	Generate_and_Concate_and_save(config, index=[29800, 39800, 21650, 48650], G=5) # 0.7480519480519481
	Generate_and_Concate_and_save(config, index=[49350, 48050, 41150, 4800], G=6)  # 0.9766233766233766

