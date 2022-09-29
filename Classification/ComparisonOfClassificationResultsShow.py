import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']

classifier  = ['KNN','SVM','DecisionTree','ResNet18']
Acc_r_r_max = list(np.loadtxt('D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\Acc_r_r_max.csv'))
Acc_t_t_max = list(np.loadtxt('D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\Acc_t_t_max.csv'))
Acc_r_t_max = list(np.loadtxt('D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\Acc_r_t_max.csv'))
Acc_t_r_max = list(np.loadtxt('D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\Acc_t_r_max.csv'))
Acc_r_enhanced_max = list(np.loadtxt('D:\\All_Datasets\\GestureClassificationResults\\MachineLearning\\Acc_r_enhanced_max.csv'))

ResNet18_r_r = np.loadtxt('D:\\All_Datasets\\GestureClassificationResults\\DeepLearning\\ResNet18_r_r.csv').tolist()
ResNet18_t_t = np.loadtxt('D:\\All_Datasets\\GestureClassificationResults\\DeepLearning\\ResNet18_t_t.csv').tolist()
ResNet18_r_t = np.loadtxt('D:\\All_Datasets\\GestureClassificationResults\\DeepLearning\\ResNet18_r_t.csv').tolist()
ResNet18_t_r = np.loadtxt('D:\\All_Datasets\\GestureClassificationResults\\DeepLearning\\ResNet18_t_r.csv').tolist()
ResNet18_r_enhanced = np.loadtxt('D:\\All_Datasets\\GestureClassificationResults\\DeepLearning\\ResNet18_r_enhanced.csv').tolist()

Acc_r_r = Acc_r_r_max + [ResNet18_r_r]
Acc_t_t = Acc_t_t_max + [ResNet18_t_t]
Acc_r_t = Acc_r_t_max + [ResNet18_r_t]
Acc_t_r = Acc_t_r_max + [ResNet18_t_r]
Acc_r_enhanced = Acc_r_enhanced_max + [ResNet18_r_enhanced]
print(Acc_r_r)
plt.figure(5,(10,7))
# plt.title('Comparison of classification accuracy',fontsize=24)
plt.bar(x=np.arange(4),height=Acc_r_r,width=0.2,label='NF-NF',edgecolor = 'white',color='darkred')
plt.bar(x=np.arange(4)+0.2,height=Acc_t_t,width=0.2,label='F-F',edgecolor = 'white',color='indianred')
plt.bar(x=np.arange(4)+0.2*2,height=Acc_r_t,width=0.2,label='NF-F',edgecolor = 'white',color='darksalmon')
plt.bar(x=np.arange(4)+0.2*3,height=Acc_t_r,width=0.2,label='F-NF',edgecolor = 'white',color='mistyrose')
plt.xticks(np.arange(4)+0.3,classifier,fontsize = 24)
plt.ylabel('Classification accuracy', fontsize=24)
plt.grid(linestyle='--')
plt.legend(loc = 'lower right',fontsize=24,framealpha=0.35)
plt.tick_params(labelsize=24)

for i in range(4):
	plt.text(x=i-0.2,y=Acc_r_r[i]+0.01,s=round(Acc_r_r[i],3),fontsize=20)
for i in range(4):
	plt.text(x=i+0.15,y=Acc_t_t[i]+0.01,s=round(Acc_t_t[i],3),fontsize=20)
for i in range(4):
	plt.text(x=i+0.05*4,y=Acc_r_t[i]+0.01,s=round(Acc_r_t[i],3),fontsize=20)
for i in range(4):
	plt.text(x=i+0.1*5,y=Acc_t_r[i]+0.01,s=round(Acc_t_r[i],3),fontsize=20)


plt.figure(6,(10,7))
# plt.title('Accuracy comparison after data enhancement',fontsize=24)
plt.bar(x=np.arange(4),height=Acc_r_enhanced,width=0.2,label='NF-Enhanced',edgecolor = 'white',color='darksalmon')

plt.xticks(np.arange(4),classifier,fontsize = 24)
plt.ylabel('Classification accuracy', fontsize=24)
plt.grid(linestyle='--')
plt.legend(loc = 'lower right',fontsize=24,framealpha=0.35)
plt.tick_params(labelsize=24)

for i in range(4):
	plt.text(x=i-0.12,y=Acc_r_enhanced[i]+0.01,s=round(Acc_r_enhanced[i],3),fontsize=20)

plt.show()