import tensorflow as tf
import os
import numpy as np
from Classification.DeepLearning.CNN_Models import ResNet18

'''
Classification prediction
'''

def predict(model_save_path,ValidationSet_path,result_save_path):

	model = ResNet18([2,2,2,2])
	model.load_weights(model_save_path)

	pathDir = os.listdir(ValidationSet_path)
	true = []
	false = []

	for i in pathDir:
		if i[-3:] == 'csv':
			emg_path = ValidationSet_path + str(i)
			# print(emg_path)
			emg = np.loadtxt(emg_path, delimiter=',')
			# emg = np.loadtxt(emg_path, delimiter=',')[:,3:4]
			# print('emg.shape',emg.shape)
			emg_predict = emg.reshape(1, emg.shape[0], emg.shape[1])
			# emg_predict = emg[512:512+1024].reshape(1, 1024, 1)
			result = model.predict(emg_predict)
			pred = tf.argmax(result, axis=1)
			np_pred = np.array(pred)

			label = int(i[17])
			if np_pred[0] == label:
				true.append(i)
			else:
				false.append(i)
				print('label:',label,'===> pred:',np_pred)
			print('len(false):', len(false) ,' Accuracy:', len(true) / len(pathDir))

	Accuracy = len(true) / len(pathDir)
	print('Accuracy:', Accuracy)
	print('len(pathDir):', len(pathDir))
	print('len(true):', len(true))
	np.savetxt(result_save_path,[Accuracy],delimiter=',')

if __name__ == '__main__':
	r_model_save_path = './SavedModels/relax/4ch/checkpoint_ResNet18/ResNet18_4ch.ckpt'
	t_model_save_path = './SavedModels/tired/4ch/checkpoint_ResNet18/ResNet18_4ch.ckpt'
	r_ValidationSet_path = 'D:\\All_Datasets\\CNNValidationSet\\relax\\'
	t_ValidationSet_path = 'D:\\All_Datasets\\CNNValidationSet\\tired\\'
	enhanced_ValidationSet_path = 'D:\\All_Datasets\\EnhancedValidationSet\\'
	r_r_result_save_path = 'D:\\All_Datasets\\GestureClassificationResults\\DeepLearning\\ResNet18_r_r.csv'
	t_t_result_save_path = 'D:\\All_Datasets\\GestureClassificationResults\\DeepLearning\\ResNet18_t_t.csv'
	r_t_result_save_path = 'D:\\All_Datasets\\GestureClassificationResults\\DeepLearning\\ResNet18_r_t.csv'
	t_r_result_save_path = 'D:\\All_Datasets\\GestureClassificationResults\\DeepLearning\\ResNet18_t_r.csv'
	r_enhanced_result_save_path = 'D:\\All_Datasets\\GestureClassificationResults\\DeepLearning\\ResNet18_r_enhanced.csv'

	predict(r_model_save_path, r_ValidationSet_path, r_r_result_save_path)
	predict(t_model_save_path, t_ValidationSet_path, t_t_result_save_path)
	predict(r_model_save_path, t_ValidationSet_path, r_t_result_save_path)
	predict(t_model_save_path, r_ValidationSet_path, t_r_result_save_path)
	predict(r_model_save_path, enhanced_ValidationSet_path, r_enhanced_result_save_path)
