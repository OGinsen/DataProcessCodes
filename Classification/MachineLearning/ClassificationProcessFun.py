import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier



def KNN(inX,dataSet,labels,K):
	'''
	K-nearest neighbor algorithm
	:param inX: Unknown test sample
	:param dataSet: Known training samples
	:param labels: Label of training sample
	:param k: Nearest K samples
	:return: Prediction type of test sample
	'''
	distance = (np.sum((dataSet-inX)**2,axis=1))**0.5 # Calculating Euler distance between unknown test sample and known training
	sortedDist = distance.argsort() # Sort the calculated Euclidean distance from small to large, and then return the corresponding index value
	classCount = {} # Used to count the number of labels (categories) of the nearest K adjacent sample data
	for k in range(K):
		votrLabel = labels[sortedDist[k]] # Obtain labels of adjacent K samples
		classCount[votrLabel] = classCount.get(votrLabel,0)+1 # Count the number of each tag
	maxType = 0
	maxCount = -1 # Assume that the initial tag is 0 and the number is -1
	for key,value in classCount.items(): # Traverse the labels and corresponding numbers in the dictionary. Key is the label and value is the corresponding number
		if value>maxCount:
			maxType=key
			maxCount = value
	return maxType

def AutoNorm(Features_DataSet):
	MinValue = np.min(Features_DataSet)
	MaxValue = np.max(Features_DataSet)
	NormFeatures = (Features_DataSet - MinValue) / (MaxValue - MinValue)
	return NormFeatures, MinValue, MaxValue

def F_test(TrainingSet_path):

	TrainingSet = pd.read_csv(TrainingSet_path).iloc[:,1:] # The first column is the row index value, which is removed

	NormTrainingSet, TrainingSet_MinValue, TrainingSet_MaxValue = AutoNorm(TrainingSet.iloc[:,:-1]) # Normalize the features. The last column is labeled and removed


	NormTrainingSet.insert(loc = len(NormTrainingSet.columns),column = 'label', # Normalized features are combined with label
											   value = TrainingSet.iloc[:,-1])

	Feature_names = NormTrainingSet.columns

	X_ = np.zeros((7,NormTrainingSet.iloc[:,:-1].shape[1]))
	SSE = np.zeros((7,NormTrainingSet.iloc[:,:-1].shape[1]))
	n = []
	for i in range(7):
		Class_index = NormTrainingSet[NormTrainingSet['label']==i].index.tolist() # Get index with label i
		n.append(len(Class_index))
		X_[i] = NormTrainingSet.loc[Class_index,Feature_names[:-1]].mean()
		SSE[i] = np.sum((NormTrainingSet.loc[Class_index,Feature_names[:-1]] - X_[i])**2,axis=0)
	X__ = NormTrainingSet.iloc[:,:-1].values.sum(axis=0) / np.sum(n)
	SSB = np.sum(np.array(n).reshape(-1,1) * (X_ - X__)**2,axis=0)
	SSE = np.sum(SSE,axis=0)
	MSB = SSB / (len(n)-1)
	MSE = SSE / (np.sum(n) - len(n))
	F = MSB / MSE

	F = pd.DataFrame(F[np.newaxis,:],columns=Feature_names[:-1])

	feature_F = []
	for index in range(0,F.shape[1],4):
		feature_F.append(F.iloc[:, index : index + 4].mean(axis=1)) # Each feature has four channels. Calculate the average value of F of the four channels of the same feature

	selected_feature_names = []
	for n in range(0,len(Feature_names)-1,4):
		selected_feature_names.append(Feature_names[n][:-3]) # Keep only the feature name and delete the channel name

	feature_F = pd.DataFrame(np.array(feature_F),index = selected_feature_names,columns=['F']) # Reconstruct the F matrix in dataframe format so that F corresponds to corresponding features

	Sorted_F = feature_F.sort_values('F',ascending=False) # Sort by column 'F', False for descending

	Selected_Features = Sorted_F.index # Get the corresponding feature name after descending arrangement

	return Selected_Features, NormTrainingSet,TrainingSet_MinValue, TrainingSet_MaxValue


def Train(TrainingSet_path,ValidationSet_path,TrainingSetType,ValidationSetType,Classifier,ResultSave_path):
	Selected_Features, NormTrainingSet,TrainingSet_MinValue, TrainingSet_MaxValue  = F_test(TrainingSet_path)

	ValidationSet = pd.read_csv(ValidationSet_path).iloc[:, 1:]
	NormValidationSet = (ValidationSet.iloc[:,:-1] - TrainingSet_MinValue) / (TrainingSet_MaxValue - TrainingSet_MinValue)
	NormValidationSet.insert(loc=len(NormValidationSet.columns), column='label',value = ValidationSet.iloc[:,-1])

	# if TrainingSetType != ValidationSetType:
	# 	ValidationSet = pd.read_csv(ValidationSet_path).iloc[:,1:]
	# 	NormValidationSet = AutoNorm(ValidationSet.iloc[:,:-1])
	# 	NormValidationSet.insert(loc = len(NormValidationSet.columns),column = 'label',
	# 											   value = ValidationSet.iloc[:,-1])

	Feature_combination = []
	Accuracy = []

	for feature in Selected_Features:
		for j in range(4):
			Feature_combination.append(feature + 'ch' + str(j))
		# G0 = NormTrainingSet[NormTrainingSet['label'] == 0].index
		# G1 = NormTrainingSet[NormTrainingSet['label'] == 1].index
		# G2 = NormTrainingSet[NormTrainingSet['label'] == 2].index
		# G3 = NormTrainingSet[NormTrainingSet['label'] == 3].index
		# G4 = NormTrainingSet[NormTrainingSet['label'] == 4].index
		# G5 = NormTrainingSet[NormTrainingSet['label'] == 5].index
		# G6 = NormTrainingSet[NormTrainingSet['label'] == 6].index
		#
		# rate = 0.9
		# arr = np.arange(len(G0))
		# np.random.shuffle(arr)
		#
		# G0Traing_index = G0[arr[:int(rate * len(G0))]].tolist()
		# G1Traing_index = G1[arr[:int(rate * len(G1))]].tolist()
		# G2Traing_index = G2[arr[:int(rate * len(G2))]].tolist()
		# G3Traing_index = G3[arr[:int(rate * len(G3))]].tolist()
		# G4Traing_index = G4[arr[:int(rate * len(G4))]].tolist()
		# G5Traing_index = G5[arr[:int(rate * len(G5))]].tolist()
		# G6Traing_index = G6[arr[:int(rate * len(G6))]].tolist()
		# AllTraining_index = G0Traing_index + G1Traing_index + G2Traing_index + G3Traing_index + G4Traing_index + G5Traing_index + G6Traing_index
		# SelectedTrainingSet = NormTrainingSet.loc[AllTraining_index, Feature_combination].values
		# SelectedTrainingLabels = NormTrainingSet.loc[AllTraining_index, 'label'].values

		SelectedTrainingSet = NormTrainingSet.loc[:, Feature_combination].values
		SelectedTrainingLabels = NormTrainingSet.loc[:, 'label'].values

		SelectedValidationSet = NormValidationSet.loc[:, Feature_combination].values
		SelectedValidationLabels = NormValidationSet.loc[:, 'label'].values

		# if TrainingSetType == ValidationSetType:
		# 	G0Validation_index = G0[arr[int(rate * len(G0))]:].tolist()
		# 	G1Validation_index = G1[arr[int(rate * len(G1))]:].tolist()
		# 	G2Validation_index = G2[arr[int(rate * len(G2))]:].tolist()
		# 	G3Validation_index = G3[arr[int(rate * len(G3))]:].tolist()
		# 	G4Validation_index = G4[arr[int(rate * len(G4))]:].tolist()
		# 	G5Validation_index = G5[arr[int(rate * len(G5))]:].tolist()
		# 	G6Validation_index = G6[arr[int(rate * len(G6))]:].tolist()
		# 	AllValidation_index = G0Validation_index + G1Validation_index + G2Validation_index + G3Validation_index + G4Validation_index + G5Validation_index + G6Validation_index

		# 	SelectedValidationSet = NormTrainingSet.loc[AllValidation_index, Feature_combination].values
		# 	SelectedValidationLabels = NormTrainingSet.loc[AllValidation_index, 'label'].values
		# else:
		# 	G0V = NormValidationSet[NormValidationSet['label'] == 0].index
		# 	G1V = NormValidationSet[NormValidationSet['label'] == 1].index
		# 	G2V = NormValidationSet[NormValidationSet['label'] == 2].index
		# 	G3V = NormValidationSet[NormValidationSet['label'] == 3].index
		# 	G4V = NormValidationSet[NormValidationSet['label'] == 4].index
		# 	G5V = NormValidationSet[NormValidationSet['label'] == 5].index
		# 	G6V = NormValidationSet[NormValidationSet['label'] == 6].index
		#
		# 	G0Validation_index = G0V[arr[int(rate * len(G0))]:].tolist()
		# 	G1Validation_index = G1V[arr[int(rate * len(G1))]:].tolist()
		# 	G2Validation_index = G2V[arr[int(rate * len(G2))]:].tolist()
		# 	G3Validation_index = G3V[arr[int(rate * len(G3))]:].tolist()
		# 	G4Validation_index = G4V[arr[int(rate * len(G4))]:].tolist()
		# 	G5Validation_index = G5V[arr[int(rate * len(G5))]:].tolist()
		# 	G6Validation_index = G6V[arr[int(rate * len(G6))]:].tolist()
		# 	AllValidation_index = G0Validation_index + G1Validation_index + G2Validation_index + G3Validation_index + G4Validation_index + G5Validation_index + G6Validation_index
		#
		# 	SelectedValidationSet = NormValidationSet.loc[AllValidation_index, Feature_combination].values
		# 	SelectedValidationLabels = NormValidationSet.loc[AllValidation_index, 'label'].values


		# Shuffle the data randomly
		SelectedTrainingSet_Size = SelectedTrainingSet.shape[0]
		Taining_arr = np.arange(SelectedTrainingSet_Size)
		np.random.shuffle(Taining_arr)
		SelectedTrainingSet = SelectedTrainingSet[Taining_arr]
		SelectedTrainingLabels = SelectedTrainingLabels[Taining_arr]

		SelectedValidationSet_Size = SelectedValidationSet.shape[0]
		Validation_arr = np.arange(SelectedValidationSet_Size)
		np.random.shuffle(Validation_arr)
		SelectedValidationSet = SelectedValidationSet[Validation_arr]
		SelectedValidationLabels = SelectedValidationLabels[Validation_arr]

		if Classifier == 'KNN':
			K = 5
			error = 0
			for n in range(len(SelectedValidationSet)):
				result = KNN(SelectedValidationSet[n, :], SelectedTrainingSet, SelectedTrainingLabels, K)
				if result != SelectedValidationLabels[n]:
					error = error + 1.0

			print('KNN-' + TrainingSetType + '-' + ValidationSetType + 'Accuracy:',1-error/len(SelectedValidationSet))
			Accuracy.append(1-error/len(SelectedValidationSet))
		elif Classifier == 'SVM' :
			model = svm.SVC(kernel='poly')
			model.fit(SelectedTrainingSet, SelectedTrainingLabels)
			error = 0
			for i in range(len(SelectedValidationSet)):
				result = model.predict(SelectedValidationSet[i:i+1, :])
				if result != SelectedValidationLabels[i]:
					error = error + 1.0
			print('SVM-' + TrainingSetType + '-' + ValidationSetType + 'Accuracy:', 1 - error / len(SelectedValidationSet))
			Accuracy.append(1 - error / len(SelectedValidationSet))

		elif Classifier == 'DecisionTree' :
			model = DecisionTreeClassifier()
			model.fit(SelectedTrainingSet, SelectedTrainingLabels)
			error = 0
			for i in range(len(SelectedValidationSet)):
				result = model.predict(SelectedValidationSet[i:i + 1, :])
				if result != SelectedValidationLabels[i]:
					error = error + 1.0
			print('DecisionTree-' + TrainingSetType + '-' + ValidationSetType + 'Accuracy:', 1 - error / len(SelectedValidationSet))
			Accuracy.append(1 - error / len(SelectedValidationSet))

	np.savetxt(ResultSave_path + 'Selected_Features.csv', Feature_combination, delimiter=',',fmt='%s')
	if Classifier == 'KNN':
		np.savetxt(ResultSave_path + 'AccuracyKNN_' + TrainingSetType + '_' + ValidationSetType + '_RawSignal.csv',
				   Accuracy, delimiter=',')
	elif Classifier =='SVM':
		np.savetxt(ResultSave_path + 'AccuracySVM_' + TrainingSetType + '_' + ValidationSetType + '_RawSignal.csv',
				   Accuracy, delimiter=',')
	elif Classifier == 'DecisionTree':
		np.savetxt(ResultSave_path + 'AccuracyDT_' + TrainingSetType + '_' + ValidationSetType + '_RawSignal.csv',
				   Accuracy, delimiter=',')



if __name__ == "__main__":
	TrainingSet_path = 'D:\\All_Datasets\\GestureClassificationFeatures\\relax\\relax_training_features.csv'
	Validation_path = 'D:\\All_Datasets\\GestureClassificationFeatures\\relax\\relax_Validation_features.csv'
	F_test(TrainingSet_path)









