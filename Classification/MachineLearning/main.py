from Classification.MachineLearning.ClassificationProcessFun import *

r_TrainingSet_path = 'D:\\All_Datasets\\GestureClassificationFeatures\\relax\\relax_training_features.csv'
r_validationSet_path = 'D:\\All_Datasets\\GestureClassificationFeatures\\relax\\relax_validation_features.csv'

t_TrainingSet_path = 'D:\\All_Datasets\\GestureClassificationFeatures\\tired\\tired_training_features.csv'
t_validationSet_path = 'D:\\All_Datasets\\GestureClassificationFeatures\\tired\\tired_validation_features.csv'

enhanced_valiadationSet_path = 'D:\\All_Datasets\\GestureClassificationFeatures\\enhanced\\enhanced_validation_features.csv'

ResultSave_path = 'D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\'

Train(r_TrainingSet_path,enhanced_valiadationSet_path,'relax','enhanced',Classifier='KNN', ResultSave_path=ResultSave_path)
Train(r_TrainingSet_path,enhanced_valiadationSet_path,'relax','enhanced',Classifier='SVM', ResultSave_path=ResultSave_path)
Train(r_TrainingSet_path,enhanced_valiadationSet_path,'relax','enhanced',Classifier='DecisionTree',ResultSave_path=ResultSave_path)


Train(r_TrainingSet_path,r_validationSet_path,'relax','relax',Classifier='KNN', ResultSave_path=ResultSave_path)
Train(r_TrainingSet_path,r_validationSet_path,'relax','relax',Classifier='SVM', ResultSave_path=ResultSave_path)
Train(r_TrainingSet_path,r_validationSet_path,'relax','relax',Classifier='DecisionTree',ResultSave_path=ResultSave_path)


Train(t_TrainingSet_path,t_validationSet_path,'tired','tired',Classifier='KNN', ResultSave_path=ResultSave_path)
Train(t_TrainingSet_path,t_validationSet_path,'tired','tired',Classifier='SVM', ResultSave_path=ResultSave_path)
Train(t_TrainingSet_path,t_validationSet_path,'tired','tired',Classifier='DecisionTree',ResultSave_path=ResultSave_path)


Train(r_TrainingSet_path,t_validationSet_path,'relax','tired',Classifier='KNN', ResultSave_path=ResultSave_path)
Train(r_TrainingSet_path,t_validationSet_path,'relax','tired',Classifier='SVM', ResultSave_path=ResultSave_path)
Train(r_TrainingSet_path,t_validationSet_path,'relax','tired',Classifier='DecisionTree',ResultSave_path=ResultSave_path)


Train(t_TrainingSet_path,r_validationSet_path,'tired','relax',Classifier='KNN', ResultSave_path=ResultSave_path)
Train(t_TrainingSet_path,r_validationSet_path,'tired','relax',Classifier='SVM', ResultSave_path=ResultSave_path)
Train(t_TrainingSet_path,r_validationSet_path,'tired','relax',Classifier='DecisionTree',ResultSave_path=ResultSave_path)

