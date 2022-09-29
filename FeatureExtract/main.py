from FeatureExtract.FeatureExtractSaveFun import FeatureExtractSave

'''
Save extracted features
'''

relax_training_path = 'D:\\All_Datasets\\GestureClassificationDatasets\\relax\\'
relax_validation_path = 'D:\\All_Datasets\\CNNValidationSet\\relax'
relax_training_features_save_path = 'D:\\All_Datasets\\GestureClassificationFeatures\\relax\\relax_training_features.csv'
relax_validation_features_save_path = 'D:\\All_Datasets\\GestureClassificationFeatures\\relax\\relax_validation_features.csv'

tired_training_path = 'D:\\All_Datasets\\GestureClassificationDatasets\\tired\\'
tired_validation_path = 'D:\\All_Datasets\\CNNValidationSet\\tired\\'
tired_training_features_save_path = 'D:\\All_Datasets\\GestureClassificationFeatures\\tired\\tired_training_features.csv'
tired_validation_features_save_path = 'D:\\All_Datasets\\GestureClassificationFeatures\\tired\\tired_validation_features.csv'

Enhanced_validation_path = 'D:\\All_Datasets\\EnhancedValidationSet\\'
Enhanced_validation_features_save_path = 'D:\\All_Datasets\\GestureClassificationFeatures\\enhanced\\enhanced_validation_features.csv'

FeatureExtractSave(data_path=relax_training_path,datasets_type='relax',features_save_psth=relax_training_features_save_path)
FeatureExtractSave(data_path=relax_validation_path,datasets_type='relax',features_save_psth=relax_validation_features_save_path)

FeatureExtractSave(data_path=tired_training_path,datasets_type='tired',features_save_psth=tired_training_features_save_path)
FeatureExtractSave(data_path=tired_validation_path,datasets_type='tired',features_save_psth=tired_validation_features_save_path)

FeatureExtractSave(data_path=Enhanced_validation_path,datasets_type='tired',features_save_psth=Enhanced_validation_features_save_path)
