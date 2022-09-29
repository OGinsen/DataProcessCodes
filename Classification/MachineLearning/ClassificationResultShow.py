import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']

KNN_r_r_path = 'D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\AccuracyKNN_relax_relax_RawSignal.csv'
SVM_r_r_path = 'D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\AccuracySVM_relax_relax_RawSignal.csv'
DT_r_r_path  = 'D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\AccuracyDT_relax_relax_RawSignal.csv'

KNN_t_t_path = 'D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\AccuracyKNN_tired_tired_RawSignal.csv'
SVM_t_t_path = 'D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\AccuracySVM_tired_tired_RawSignal.csv'
DT_t_t_path  = 'D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\AccuracyDT_tired_tired_RawSignal.csv'

KNN_r_t_path = 'D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\AccuracyKNN_relax_tired_RawSignal.csv'
SVM_r_t_path = 'D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\AccuracySVM_relax_tired_RawSignal.csv'
DT_r_t_path  = 'D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\AccuracyDT_relax_tired_RawSignal.csv'

KNN_t_r_path = 'D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\AccuracyKNN_tired_relax_RawSignal.csv'
SVM_t_r_path = 'D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\AccuracySVM_tired_relax_RawSignal.csv'
DT_t_r_path  = 'D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\AccuracyDT_tired_relax_RawSignal.csv'

KNN_r_enhanced_path = 'D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\AccuracyKNN_relax_enhanced_RawSignal.csv'
SVM_r_enhanced_path = 'D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\AccuracySVM_relax_enhanced_RawSignal.csv'
DT_r_enhanced_path  = 'D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\AccuracyDT_relax_enhanced_RawSignal.csv'

Acc_KNN_r_r = np.loadtxt(KNN_r_r_path,delimiter=',')
Acc_SVM_r_r = np.loadtxt(SVM_r_r_path,delimiter=',')
Acc_DT_r_r  = np.loadtxt(DT_r_r_path,delimiter=',')
print('Acc_KNN_r_r Max:',max(Acc_KNN_r_r))
print('Acc_SVM_r_r Max:',max(Acc_SVM_r_r))
print('Acc_DT_r_r Max:', max(Acc_DT_r_r))

Acc_KNN_t_t = np.loadtxt(KNN_t_t_path,delimiter=',')
Acc_SVM_t_t = np.loadtxt(SVM_t_t_path,delimiter=',')
Acc_DT_t_t  = np.loadtxt(DT_t_t_path,delimiter=',')
print('Acc_KNN_t_t Max:',max(Acc_KNN_t_t))
print('Acc_SVM_t_t Max:',max(Acc_SVM_t_t))
print('Acc_DT_t_t Max:', max(Acc_DT_t_t))

Acc_KNN_r_t = np.loadtxt(KNN_r_t_path,delimiter=',')
Acc_SVM_r_t = np.loadtxt(SVM_r_t_path,delimiter=',')
Acc_DT_r_t  = np.loadtxt(DT_r_t_path,delimiter=',')
print('Acc_KNN_r_t Max:',max(Acc_KNN_r_t))
print('Acc_SVM_r_t Max:',max(Acc_SVM_r_t))
print('Acc_DT_r_t Max:', max(Acc_DT_r_t))

Acc_KNN_t_r = np.loadtxt(KNN_t_r_path,delimiter=',')
Acc_SVM_t_r = np.loadtxt(SVM_t_r_path,delimiter=',')
Acc_DT_t_r  = np.loadtxt(DT_t_r_path,delimiter=',')
print('Acc_KNN_t_r Max:',max(Acc_KNN_t_r))
print('Acc_SVM_t_r Max:',max(Acc_SVM_t_r))
print('Acc_DT_t_r Max:', max(Acc_DT_t_r))

Acc_KNN_r_enhanced = np.loadtxt(KNN_r_enhanced_path,delimiter=',')
Acc_SVM_r_enhanced = np.loadtxt(SVM_r_enhanced_path,delimiter=',')
Acc_DT_r_enhanced  = np.loadtxt(DT_r_enhanced_path,delimiter=',')
print('Acc_KNN_r_enhanced Max:',max(Acc_KNN_r_enhanced))
print('Acc_SVM_r_enhanced Max:',max(Acc_SVM_r_enhanced))
print('Acc_DT_r_enhanced Max:', max(Acc_DT_r_enhanced))

plt.figure(1,figsize=(10,7))
# plt.title('Classification accuracy when both training and test sets are non-fatigue data',fontsize=24)
plt.plot(range(1,53),Acc_KNN_r_r,'r-*',label='KNN-NF-NF')
plt.plot(range(1,53),Acc_SVM_r_r,'g-s',label='SVM-NF-NF')
plt.plot(range(1,53),Acc_DT_r_r,'b-+',label='DecisionTree-NF-NF')
plt.ylabel('Classification accuracy', fontsize=24)
plt.xlabel('Number of features selected', fontsize=24)
plt.legend(prop = {'size':24})
plt.tick_params(labelsize=24)
plt.grid(linestyle='--')

plt.figure(2,figsize=(10,7))
# plt.title('Classification accuracy when both training and test sets are fatigue data',fontsize=24)
plt.plot(range(1,53),Acc_KNN_t_t,'r-*',label='KNN-F-F')
plt.plot(range(1,53),Acc_SVM_t_t,'g-s',label='SVM-F-F')
plt.plot(range(1,53),Acc_DT_t_t,'b-+',label='DecisionTree-F-F')
plt.ylabel('Classification accuracy', fontsize=24)
plt.xlabel('Number of features selected', fontsize=24)
plt.legend(prop = {'size':24})
plt.tick_params(labelsize=24)
plt.grid(linestyle='--')


plt.figure(3,figsize=(10,7))
# plt.title('Classification accuracy when the training set is non-fatigue data and the test set is fatigue data',fontsize=24)
plt.plot(range(1,53),Acc_KNN_r_t,'r-*',label='KNN-NF-F')
plt.plot(range(1,53),Acc_SVM_r_t,'g-s',label='SVM-NF-F')
plt.plot(range(1,53),Acc_DT_r_t,'b-+',label='DecisionTree-NF-F')
plt.ylabel('Classification accuracy', fontsize=24)
plt.xlabel('Number of features selected', fontsize=24)
plt.legend(prop = {'size':24})
plt.tick_params(labelsize=24)
plt.grid(linestyle='--')


plt.figure(4,figsize=(10,7))
# plt.title('Classification accuracy when the training set is fatigue data and the test set is non-fatigue data',fontsize=24)
plt.plot(range(1,53),Acc_KNN_t_r,'r-*',label='KNN-F-NF')
plt.plot(range(1,53),Acc_SVM_t_r,'g-s',label='SVM-F-NF')
plt.plot(range(1,53),Acc_DT_t_r,'b-+',label='DecisionTree-F-NF')
plt.ylabel('Classification accuracy', fontsize=24)
plt.xlabel('Number of features selected', fontsize=24)
plt.legend(prop = {'size':24})
plt.tick_params(labelsize=24)
plt.grid(linestyle='--')

plt.figure(6,figsize=(10,7))
# plt.title('Classification accuracy when the training set is non-fatigue data and the test set is enhanced data',fontsize=24)
plt.plot(range(1,53),Acc_KNN_r_enhanced,'r-*',label='KNN-NF-Enhanced')
plt.plot(range(1,53),Acc_SVM_r_enhanced,'g-s',label='SVM-NF-Enhanced')
plt.plot(range(1,53),Acc_DT_r_enhanced,'b-+',label='DecisionTree-NF-Enhanced')
plt.ylabel('Classification accuracy', fontsize=24)
plt.xlabel('Number of features selected', fontsize=24)
plt.legend(prop = {'size':24})
plt.tick_params(labelsize=24)
plt.grid(linestyle='--')

classifier  = ['KNN','SVM','DecisionTree']
Acc_r_r_max = [max(Acc_KNN_r_r),max(Acc_SVM_r_r),max(Acc_DT_r_r)]
Acc_t_t_max = [max(Acc_KNN_t_t),max(Acc_SVM_t_t),max(Acc_DT_t_t)]
Acc_r_t_max = [max(Acc_KNN_r_t),max(Acc_SVM_r_t),max(Acc_DT_r_t)]
Acc_t_r_max = [max(Acc_KNN_t_r),max(Acc_SVM_t_r),max(Acc_DT_t_r)]
Acc_r_enhanced_max = [max(Acc_KNN_r_enhanced),max(Acc_SVM_r_enhanced),max(Acc_DT_r_enhanced)]

np.savetxt('D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\Acc_r_r_max.csv',Acc_r_r_max)
np.savetxt('D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\Acc_t_t_max.csv',Acc_t_t_max)
np.savetxt('D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\Acc_r_t_max.csv',Acc_r_t_max)
np.savetxt('D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\Acc_t_r_max.csv',Acc_t_r_max)
np.savetxt('D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\Acc_r_enhanced_max.csv',Acc_r_enhanced_max)

plt.figure(5,(10,7))
# plt.title('Comparison of classification accuracy',fontsize=24)
plt.bar(x=np.arange(3),height=Acc_r_r_max,width=0.15,label='NF-NF',edgecolor = 'white',color='darkred')
plt.bar(x=np.arange(3)+0.15,height=Acc_t_t_max,width=0.15,label='F-F',edgecolor = 'white',color='indianred')
plt.bar(x=np.arange(3)+0.15*2,height=Acc_r_t_max,width=0.15,label='NF-F',edgecolor = 'white',color='darksalmon')
plt.bar(x=np.arange(3)+0.15*3,height=Acc_t_r_max,width=0.15,label='F-NF',edgecolor = 'white',color='mistyrose')
plt.xticks(np.arange(3)+0.2,classifier,fontsize = 24)
plt.ylabel('Classification accuracy', fontsize=24)
plt.grid(linestyle='--')
plt.legend(loc = 'lower right',fontsize=24,framealpha=0.3)
plt.tick_params(labelsize=24)

for i in range(3):
	plt.text(x=i-0.15,y=Acc_r_r_max[i]+0.01,s=round(Acc_r_r_max[i],3),fontsize=24)
for i in range(3):
	plt.text(x=i+0.05,y=Acc_t_t_max[i]+0.01,s=round(Acc_t_t_max[i],3),fontsize=24)
for i in range(3):
	plt.text(x=i+0.05*3,y=Acc_r_t_max[i]+0.01,s=round(Acc_r_t_max[i],3),fontsize=24)
for i in range(3):
	plt.text(x=i+0.1*4,y=Acc_t_r_max[i]+0.01,s=round(Acc_t_r_max[i],3),fontsize=24)
plt.show()

