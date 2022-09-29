import numpy as np
import os

def GeneratedLabel_txt(DataSet_path,txt_path,Gesture_type:str):
	txt = open(txt_path, 'w')
	for x, ys ,zs in os.walk(DataSet_path, 'r'):
		for z in zs:
			if z[-3:]=='csv' and z[11:18] == Gesture_type:
			# if z[-3:] == 'txt' and z[5:10] == Gesture_type:
				emg_file = x + '\\' + z
				print(emg_file)
				label = z[17]
				# label = z[11]
				txt.write(emg_file)
				txt.write(' ')
				txt.write(label)
				txt.write('\n')
	txt.close()

def GenerateDataSet(txt_path,SaveDataSet_path,SaveLabel_path,AllS_rest_path):
	file = open(txt_path,'r') # Open txt file as read-only
	contents = file.readlines() # Read all lines in the file
	file.close() # Close file
	x , y_ = [] , [] # Create an empty list
	for content in contents: # Row by row extraction
		value = content.split() # Split with spaces
		datasets_path = value[0] # Concatenate file paths and file names


		rest_path = AllS_rest_path[int(datasets_path.split('\\')[-1][1])-1]
		# rest_path = AllS_rest_path[int(datasets_path.split('\\')[3][1]) - 1]

		rest_emg = np.loadtxt(rest_path, delimiter=',')

		datasets = np.loadtxt(datasets_path, delimiter=',')
		# datasets = np.concatenate((rest_emg[:int((2**14-len(datasets))/2)+1,:],datasets,
		# 						   rest_emg[int((2**14-len(datasets))/2)+1:
		# 								    int((2**14-len(datasets))/2)+1+(2**14-(int((2**14-len(datasets))/2)+1)-len(datasets))
		# 								:]),axis=0)
		datasets = np.concatenate((rest_emg[:512, :], datasets, rest_emg[-512:, :]), axis=0)
		print('datasets.shape:',datasets.shape)
		x.append(datasets)
		y_.append(value[1])
		print('loading:' + content) # Print status prompt
	x = np.array(x)
	print('x.shape',x.shape)
	y_ = np.array(y_)
	y_ = y_.astype(np.int64) # Change to 64 bit integer
	print('x.shape, y_.shape:',x.shape, y_.shape)
	np.save(SaveDataSet_path,x)
	np.save(SaveLabel_path,y_)

if __name__ == "__main__":
	# S1_r_DataSet_path = 'D:\\All_Datasets\\ActiveDatasets\\S1\\relax'
	# S1_r_txt_path = 'D:\\PycharmProjects\\DataProcessCodes\\WGAN\\ActiveDatasetsPadded\\relax\\S1\\S1_relax_Label_AllG.txt'
	# GeneratedLabel_txt(S1_r_DataSet_path,S1_r_txt_path)
	# S1_r_SaveDataSet_path = 'D:\\PycharmProjects\\DataProcessCodes\\WGAN\\ActiveDatasetsPadded\\relax\\S1\\S1_r_ActiveDatasetsPadded_AllG.npy'
	# S1_r_SaveLabel_path = 'D:\\PycharmProjects\\DataProcessCodes\\WGAN\\ActiveDatasetsPadded\\relax\\S1\\S1_r_ActiveDatasetsPadded_AllG_Label.npy'
	# S1_r_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S1\\relax\\norm_relax_rest.txt'
	# GenerateDataSet(S1_r_txt_path,S1_r_SaveDataSet_path,S1_r_SaveLabel_path,S1_r_rest_path)
	#
	# S1_t_DataSet_path = 'D:\\All_Datasets\\ActiveDatasets\\S1\\tired'
	# S1_t_txt_path = 'D:\\PycharmProjects\\DataProcessCodes\\WGAN\\ActiveDatasetsPadded\\tired\\S1\\S1_tired_Label_AllG.txt'
	# GeneratedLabel_txt(S1_t_DataSet_path,S1_t_txt_path)
	# S1_t_SaveDataSet_path = 'D:\\PycharmProjects\\DataProcessCodes\\WGAN\\ActiveDatasetsPadded\\tired\\S1\\S1_t_ActiveDatasetsPadded_AllG.npy'
	# S1_t_SaveLabel_path = 'D:\\PycharmProjects\\DataProcessCodes\\WGAN\\ActiveDatasetsPadded\\tired\\S1\\S1_t_ActiveDatasetsPadded_AllG_Label.npy'
	# S1_t_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S1\\tired\\norm_tired_rest.txt'
	# GenerateDataSet(S1_t_txt_path,S1_t_SaveDataSet_path,S1_t_SaveLabel_path,S1_t_rest_path)

	S1_r_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S1\\relax\\norm_relax_rest.txt'
	S2_r_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S2\\relax\\norm_relax_rest.txt'
	S3_r_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S3\\relax\\norm_relax_rest.txt'
	S4_r_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S4\\relax\\norm_relax_rest.txt'
	S5_r_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S5\\relax\\norm_relax_rest.txt'
	S6_r_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S6\\relax\\norm_relax_rest.txt'
	S7_r_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S7\\relax\\norm_relax_rest.txt'
	S8_r_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S8\\relax\\norm_relax_rest.txt'
	All_r_rest_path = [S1_r_rest_path, S2_r_rest_path, S3_r_rest_path, S4_r_rest_path,
					   S5_r_rest_path, S6_r_rest_path, S7_r_rest_path, S8_r_rest_path]

	S1_t_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S1\\tired\\norm_tired_rest.txt'
	S2_t_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S2\\tired\\norm_tired_rest.txt'
	S3_t_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S3\\tired\\norm_tired_rest.txt'
	S4_t_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S4\\tired\\norm_tired_rest.txt'
	S5_t_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S5\\tired\\norm_tired_rest.txt'
	S6_t_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S6\\tired\\norm_tired_rest.txt'
	S7_t_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S7\\tired\\norm_tired_rest.txt'
	S8_t_rest_path = 'D:\\All_Datasets\\Datasets_Norm\\S8\\tired\\norm_tired_rest.txt'
	All_t_rest_path = [S1_t_rest_path, S2_t_rest_path, S3_t_rest_path, S4_t_rest_path,
					   S5_t_rest_path, S6_t_rest_path, S7_t_rest_path, S8_t_rest_path]

	for G in range(7):
		# r_DataSet_path = 'D:\\All_Datasets\\GestureClassificationDatasets\\relax'
		# r_txt_path = 'ActiveDatasetsWindowsPadded\\relax\\AllS_relax_Label_G' + str(G) + '.txt'
		# GeneratedLabel_txt(r_DataSet_path,r_txt_path,'relax_' + str(G))
		#
		# r_SaveDataSet_path = 'ActiveDatasetsWindowsPadded\\relax\\AllS_r_ActiveDatasetsWindowsPadded_G' + str(G) + '.npy'
		# r_SaveLabel_path = 'ActiveDatasetsWindowsPadded\\relax\\AllS_r_ActiveDatasetsWindowsPadded_G' + str(G) + '_Label.npy'
		# GenerateDataSet(r_txt_path,r_SaveDataSet_path,r_SaveLabel_path,All_r_rest_path)
		#
		#
		# t_DataSet_path = 'D:\\All_Datasets\\GestureClassificationDatasets\\tired'
		# t_txt_path = 'ActiveDatasetsWindowsPadded\\tired\\AllS_tired_Label_G' + str(G) + '.txt'
		# GeneratedLabel_txt(t_DataSet_path,t_txt_path,'tired_' + str(G))
		#
		# t_SaveDataSet_path = 'ActiveDatasetsWindowsPadded\\tired\\AllS_t_ActiveDatasetsWindowsPadded_G' + str(G) + '.npy'
		# t_SaveLabel_path = 'ActiveDatasetsWindowsPadded\\tired\\AllS_t_ActiveDatasetsWindowsPadded_G' + str(G) + '_Label.npy'
		# GenerateDataSet(t_txt_path,t_SaveDataSet_path,t_SaveLabel_path,All_t_rest_path)

		t_ValidationSet_path = 'D:\\All_Datasets\\CNNValidationSet\\tired'
		t_ValidationSet_txt_path = 'ActiveDatasetsWindowsPaddedValidationSet\\tired\\AllS_tired_Label_G' + str(G) + '.txt'
		GeneratedLabel_txt(t_ValidationSet_path,t_ValidationSet_txt_path,'tired_' + str(G))

		t_SaveValidationSet_path = 'ActiveDatasetsWindowsPaddedValidationSet\\tired\\AllS_t_ActiveDatasetsWindowsPaddedValidationSet_G' + str(G) + '.npy'
		t_SaveValidationSetLabel_path = 'ActiveDatasetsWindowsPaddedValidationSet\\tired\\AllS_t_ActiveDatasetsWindowsPaddedValidationSet_G' + str(G) + '_Label.npy'
		GenerateDataSet(t_ValidationSet_txt_path,t_SaveValidationSet_path,t_SaveValidationSetLabel_path,All_t_rest_path)




#########################################################################
	# r_DataSet_path = 'D:\\All_Datasets\\ActiveDatasets'
	# r_txt_path = 'D:\\PycharmProjects\\DataProcessCodes\\WGAN\\ActiveDatasetsPadded\\relax\\AllS_relax_Label_AllG.txt'
	# GeneratedLabel_txt(r_DataSet_path, r_txt_path, 'relax')
	#
	# r_SaveDataSet_path = 'D:\\PycharmProjects\\DataProcessCodes\\WGAN\\ActiveDatasetsPadded\\relax\\AllS_r_ActiveDatasetsPadded_AllG.npy'
	# r_SaveLabel_path = 'D:\\PycharmProjects\\DataProcessCodes\\WGAN\\ActiveDatasetsPadded\\relax\\AllS_r_ActiveDatasetsPadded_AllG_Label.npy'
	# GenerateDataSet(r_txt_path, r_SaveDataSet_path, r_SaveLabel_path, All_r_rest_path)
	#
	# t_DataSet_path = 'D:\\All_Datasets\\ActiveDatasets'
	# t_txt_path = 'D:\\PycharmProjects\\DataProcessCodes\\WGAN\\ActiveDatasetsPadded\\tired\\AllS_tired_Label_AllG.txt'
	# GeneratedLabel_txt(t_DataSet_path, t_txt_path, 'tired')
	#
	# t_SaveDataSet_path = 'D:\\PycharmProjects\\DataProcessCodes\\WGAN\\ActiveDatasetsPadded\\tired\\AllS_t_ActiveDatasetsPadded_AllG.npy'
	# t_SaveLabel_path = 'D:\\PycharmProjects\\DataProcessCodes\\WGAN\\ActiveDatasetsPadded\\tired\\AllS_t_ActiveDatasetsPadded_AllG_Label.npy'
	# GenerateDataSet(t_txt_path, t_SaveDataSet_path, t_SaveLabel_path, All_t_rest_path)