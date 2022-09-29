import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',size = 35)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
def FeatureShow(R_G0_feature_1,R_G0_feature_2,T_G0_feature_1,T_G0_feature_2,
				R_G1_feature_1,R_G1_feature_2,T_G1_feature_1,T_G1_feature_2,
				R_G2_feature_1,R_G2_feature_2,T_G2_feature_1,T_G2_feature_2,
				R_G3_feature_1,R_G3_feature_2,T_G3_feature_1,T_G3_feature_2,
				R_G4_feature_1,R_G4_feature_2,T_G4_feature_1,T_G4_feature_2,
				R_G5_feature_1,R_G5_feature_2,T_G5_feature_1,T_G5_feature_2,
				R_G6_feature_1,R_G6_feature_2,T_G6_feature_1,T_G6_feature_2,
				feature_1_name,feature_2_name,channel):
	lenth = min(
				len(R_G0_feature_1), len(R_G0_feature_2), len(T_G0_feature_1), len(T_G0_feature_2),
				len(R_G1_feature_1), len(R_G1_feature_2), len(T_G1_feature_1), len(T_G1_feature_2),
				len(R_G2_feature_1), len(R_G2_feature_2), len(T_G2_feature_1), len(T_G2_feature_2),
				len(R_G3_feature_1), len(R_G3_feature_2), len(T_G3_feature_1), len(T_G3_feature_2),
				len(R_G4_feature_1), len(R_G4_feature_2), len(T_G4_feature_1), len(T_G4_feature_2),
				len(R_G5_feature_1), len(R_G5_feature_2), len(T_G5_feature_1), len(T_G5_feature_2),
				len(R_G6_feature_1), len(R_G6_feature_2), len(T_G6_feature_1), len(T_G6_feature_2),
				)
	r_G0_feature_1 = R_G0_feature_1[:lenth]
	r_G0_feature_2 = R_G0_feature_2[:lenth]
	t_G0_feature_1 = T_G0_feature_1[:lenth]
	t_G0_feature_2 = T_G0_feature_2[:lenth]
	r_G1_feature_1 = R_G1_feature_1[:lenth]
	r_G1_feature_2 = R_G1_feature_2[:lenth]
	t_G1_feature_1 = T_G1_feature_1[:lenth]
	t_G1_feature_2 = T_G1_feature_2[:lenth]
	r_G2_feature_1 = R_G2_feature_1[:lenth]
	r_G2_feature_2 = R_G2_feature_2[:lenth]
	t_G2_feature_1 = T_G2_feature_1[:lenth]
	t_G2_feature_2 = T_G2_feature_2[:lenth]
	r_G3_feature_1 = R_G3_feature_1[:lenth]
	r_G3_feature_2 = R_G3_feature_2[:lenth]
	t_G3_feature_1 = T_G3_feature_1[:lenth]
	t_G3_feature_2 = T_G3_feature_2[:lenth]
	r_G4_feature_1 = R_G4_feature_1[:lenth]
	r_G4_feature_2 = R_G4_feature_2[:lenth]
	t_G4_feature_1 = T_G4_feature_1[:lenth]
	t_G4_feature_2 = T_G4_feature_2[:lenth]
	r_G5_feature_1 = R_G5_feature_1[:lenth]
	r_G5_feature_2 = R_G5_feature_2[:lenth]
	t_G5_feature_1 = T_G5_feature_1[:lenth]
	t_G5_feature_2 = T_G5_feature_2[:lenth]
	r_G6_feature_1 = R_G6_feature_1[:lenth]
	r_G6_feature_2 = R_G6_feature_2[:lenth]
	t_G6_feature_1 = T_G6_feature_1[:lenth]
	t_G6_feature_2 = T_G6_feature_2[:lenth]
	plt.figure(1,figsize=(10,8))
	plt.title('Feature comparison', fontsize=35)
	plt.scatter(r_G0_feature_1, r_G0_feature_2, c='b',marker='.', label='NF-G0-ch' + str(channel), )
	plt.scatter(t_G0_feature_1, t_G0_feature_2, c='r',marker='.', label='F -G0-ch' + str(channel), )
	plt.scatter(r_G1_feature_1, r_G1_feature_2, c='b',marker='+', label='NF-G1-ch' + str(channel), )
	plt.scatter(t_G1_feature_1, t_G1_feature_2, c='r',marker='+', label='F -G1-ch' + str(channel), )
	plt.scatter(r_G2_feature_1, r_G2_feature_2, c='b',marker='*', label='NF-G2-ch' + str(channel), )
	plt.scatter(t_G2_feature_1, t_G2_feature_2, c='r',marker='*', label='F -G2-ch' + str(channel), )
	plt.scatter(r_G3_feature_1, r_G3_feature_2, c='b',marker='s', label='NF-G3-ch' + str(channel), )
	plt.scatter(t_G3_feature_1, t_G3_feature_2, c='r',marker='s', label='F -G3-ch' + str(channel), )
	plt.scatter(r_G4_feature_1, r_G4_feature_2, c='b',marker='x', label='NF-G4-ch' + str(channel), )
	plt.scatter(t_G4_feature_1, t_G4_feature_2, c='r',marker='x', label='F -G4-ch' + str(channel), )
	plt.scatter(r_G5_feature_1, r_G5_feature_2, c='b',marker='v', label='NF-G5-ch' + str(channel), )
	plt.scatter(t_G5_feature_1, t_G5_feature_2, c='r',marker='v', label='F -G5-ch' + str(channel), )
	plt.scatter(r_G6_feature_1, r_G6_feature_2, c='b',marker='D', label='NF-G6-ch' + str(channel), )
	plt.scatter(t_G6_feature_1, t_G6_feature_2, c='r',marker='D', label='F -G6-ch' + str(channel), )
	plt.xlabel(feature_1_name, fontsize=35)
	plt.ylabel(feature_2_name, fontsize=35)
	plt.tick_params(labelsize=35)
	plt.grid(linestyle='--')
	plt.legend(fontsize=20)
	plt.show()

iEMG_r_G0 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'iEMG_r_G0.csv',delimiter=',')
# RMS_r = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'RMS_r.csv',delimiter=',')
MPF_r_G0 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MPF_r_G0.csv',delimiter=',')
# MF_r = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MF_r.csv',delimiter=',')
iEMG_t_G0 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'iEMG_t_G0.csv',delimiter=',')
# RMS_t = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'RMS_t.csv',delimiter=',')
MPF_t_G0 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MPF_t_G0.csv',delimiter=',')
# MF_t = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MF_t.csv',delimiter=',')

iEMG_r_G1 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'iEMG_r_G1.csv',delimiter=',')
MPF_r_G1 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MPF_r_G1.csv',delimiter=',')
iEMG_t_G1 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'iEMG_t_G1.csv',delimiter=',')
MPF_t_G1 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MPF_t_G1.csv',delimiter=',')

iEMG_r_G2 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'iEMG_r_G2.csv',delimiter=',')
MPF_r_G2 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MPF_r_G2.csv',delimiter=',')
iEMG_t_G2 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'iEMG_t_G2.csv',delimiter=',')
MPF_t_G2 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MPF_t_G2.csv',delimiter=',')

iEMG_r_G3 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'iEMG_r_G3.csv',delimiter=',')
MPF_r_G3 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MPF_r_G3.csv',delimiter=',')
iEMG_t_G3 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'iEMG_t_G3.csv',delimiter=',')
MPF_t_G3 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MPF_t_G3.csv',delimiter=',')

iEMG_r_G4 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'iEMG_r_G4.csv',delimiter=',')
MPF_r_G4 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MPF_r_G4.csv',delimiter=',')
iEMG_t_G4 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'iEMG_t_G4.csv',delimiter=',')
MPF_t_G4 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MPF_t_G4.csv',delimiter=',')

iEMG_r_G5 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'iEMG_r_G5.csv',delimiter=',')
MPF_r_G5 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MPF_r_G5.csv',delimiter=',')
iEMG_t_G5 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'iEMG_t_G5.csv',delimiter=',')
MPF_t_G5 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MPF_t_G5.csv',delimiter=',')

iEMG_r_G6 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'iEMG_r_G6.csv',delimiter=',')
MPF_r_G6 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MPF_r_G6.csv',delimiter=',')
iEMG_t_G6 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'iEMG_t_G6.csv',delimiter=',')
MPF_t_G6 = np.loadtxt('D:\\All_Datasets\\ActiveDatasetsFeatureShow\\' + 'MPF_t_G6.csv',delimiter=',')

print(iEMG_r_G0.shape)
lenth = min(
	len(iEMG_r_G0), len(MPF_r_G0), len(iEMG_t_G0), len(MPF_t_G0),
	len(iEMG_r_G1), len(MPF_r_G1), len(iEMG_t_G1), len(MPF_t_G1),
	len(iEMG_r_G2), len(MPF_r_G2), len(iEMG_t_G2), len(MPF_t_G2),
	len(iEMG_r_G3), len(MPF_r_G3), len(iEMG_t_G3), len(MPF_t_G3),
	len(iEMG_r_G4), len(MPF_r_G4), len(iEMG_t_G4), len(MPF_t_G4),
	len(iEMG_r_G5), len(MPF_r_G5), len(iEMG_t_G5), len(MPF_t_G5),
	len(iEMG_r_G6), len(MPF_r_G6), len(iEMG_t_G6), len(MPF_t_G6),
)
iEMG_r_G0 = iEMG_r_G0[:lenth,:]
MPF_r_G0 = MPF_r_G0[:lenth,:]
iEMG_t_G0 = iEMG_t_G0[:lenth,:]
MPF_t_G0 = MPF_t_G0[:lenth,:]
iEMG_r_G1 = iEMG_r_G1[:lenth,:]
MPF_r_G1 = MPF_r_G1[:lenth,:]
iEMG_t_G1 = iEMG_t_G1[:lenth,:]
MPF_t_G1 = MPF_t_G1[:lenth,:]
iEMG_r_G2 = iEMG_r_G2[:lenth,:]
MPF_r_G2 = MPF_r_G2[:lenth,:]
iEMG_t_G2 = iEMG_t_G2[:lenth,:]
MPF_t_G2 = MPF_t_G2[:lenth,:]
iEMG_r_G3 = iEMG_r_G3[:lenth,:]
MPF_r_G3 = MPF_r_G3[:lenth,:]
iEMG_t_G3 = iEMG_t_G3[:lenth,:]
MPF_t_G3 = MPF_t_G3[:lenth,:]
iEMG_r_G4 = iEMG_r_G4[:lenth,:]
MPF_r_G4 = MPF_r_G4[:lenth,:]
iEMG_t_G4 = iEMG_t_G4[:lenth,:]
MPF_t_G4 = MPF_t_G4[:lenth,:]
iEMG_r_G5 = iEMG_r_G5[:lenth,:]
MPF_r_G5 = MPF_r_G5[:lenth,:]
iEMG_t_G5 = iEMG_t_G5[:lenth,:]
MPF_t_G5 = MPF_t_G5[:lenth,:]
iEMG_r_G6 = iEMG_r_G6[:lenth,:]
MPF_r_G6 = MPF_r_G6[:lenth,:]
iEMG_t_G6 = iEMG_t_G6[:lenth,:]
MPF_t_G6 = MPF_t_G6[:lenth,:]

for ch in range(4):
	FeatureShow(
				iEMG_r_G0[:, ch], MPF_r_G0[:, ch], iEMG_t_G0[:, ch], MPF_t_G0[:, ch],
				iEMG_r_G1[:, ch], MPF_r_G1[:, ch], iEMG_t_G1[:, ch], MPF_t_G1[:, ch],
				iEMG_r_G2[:, ch], MPF_r_G2[:, ch], iEMG_t_G2[:, ch], MPF_t_G2[:, ch],
				iEMG_r_G3[:, ch], MPF_r_G3[:, ch], iEMG_t_G3[:, ch], MPF_t_G3[:, ch],
				iEMG_r_G4[:, ch], MPF_r_G4[:, ch], iEMG_t_G4[:, ch], MPF_t_G4[:, ch],
				iEMG_r_G5[:, ch], MPF_r_G5[:, ch], iEMG_t_G5[:, ch], MPF_t_G5[:, ch],
				iEMG_r_G6[:, ch], MPF_r_G6[:, ch], iEMG_t_G6[:, ch], MPF_t_G6[:, ch],
				'iEMG','MNF',channel=ch)

	plt.figure(2)
	plt.suptitle('IEMG feature comparison',fontsize=20)
	plt.plot(iEMG_r_G0[:, ch], 'b.', label='NF-iEMG-G0-ch' + str(ch))
	plt.plot(iEMG_t_G0[:, ch], 'r.', label='F -iEMG-G0-ch' + str(ch))
	plt.plot(iEMG_r_G1[:, ch], 'b+', label='NF-iEMG-G1-ch' + str(ch))
	plt.plot(iEMG_t_G1[:, ch], 'r+', label='F -iEMG-G1-ch' + str(ch))
	plt.plot(iEMG_r_G2[:, ch], 'b*', label='NF-iEMG-G2-ch' + str(ch))
	plt.plot(iEMG_t_G2[:, ch], 'r*', label='F -iEMG-G2-ch' + str(ch))
	plt.plot(iEMG_r_G3[:, ch], 'bs', label='NF-iEMG-G3-ch' + str(ch))
	plt.plot(iEMG_t_G3[:, ch], 'rs', label='F -iEMG-G3-ch' + str(ch))
	plt.plot(iEMG_r_G4[:, ch], 'bx', label='NF-iEMG-G4-ch' + str(ch))
	plt.plot(iEMG_t_G4[:, ch], 'rx', label='F -iEMG-G4-ch' + str(ch))
	plt.plot(iEMG_r_G5[:, ch], 'bv', label='NF-iEMG-G5-ch' + str(ch))
	plt.plot(iEMG_t_G5[:, ch], 'rv', label='F -iEMG-G5-ch' + str(ch))
	plt.plot(iEMG_r_G6[:, ch], 'bD', label='NF-iEMG-G6-ch' + str(ch))
	plt.plot(iEMG_t_G6[:, ch], 'rD', label='F -iEMG-G6-ch' + str(ch))
	plt.ylabel('IEMG characteristic value', fontsize=15)
	plt.xlabel('Number of samples', fontsize=15)
	plt.legend()
	plt.grid()

	plt.figure(3)
	plt.suptitle('MPF feature comparison', fontsize=20)
	plt.plot(MPF_r_G0[:, ch], 'b.', label='NF-MPF-G0-ch' + str(ch))
	plt.plot(MPF_t_G0[:, ch], 'r.', label='F -MPF-G0-ch' + str(ch))
	plt.plot(MPF_r_G1[:, ch], 'b+', label='NF-MPF-G1-ch' + str(ch))
	plt.plot(MPF_t_G1[:, ch], 'r+', label='F -MPF-G1-ch' + str(ch))
	plt.plot(MPF_r_G2[:, ch], 'b*', label='NF-MPF-G2-ch' + str(ch))
	plt.plot(MPF_t_G2[:, ch], 'r*', label='F -MPF-G2-ch' + str(ch))
	plt.plot(MPF_r_G3[:, ch], 'bs', label='NF-MPF-G3-ch' + str(ch))
	plt.plot(MPF_t_G3[:, ch], 'rs', label='F -MPF-G3-ch' + str(ch))
	plt.plot(MPF_r_G4[:, ch], 'bx', label='NF-MPF-G4-ch' + str(ch))
	plt.plot(MPF_t_G4[:, ch], 'rx', label='F -MPF-G4-ch' + str(ch))
	plt.plot(MPF_r_G5[:, ch], 'bv', label='NF-MPF-G5-ch' + str(ch))
	plt.plot(MPF_t_G5[:, ch], 'rv', label='F -MPF-G5-ch' + str(ch))
	plt.plot(MPF_r_G6[:, ch], 'bD', label='NF-MPF-G6-ch' + str(ch))
	plt.plot(MPF_t_G6[:, ch], 'rD', label='F -MPF-G6-ch' + str(ch))
	plt.ylabel('MPF characteristic value', fontsize=15)
	plt.xlabel('Number of samples', fontsize=15)
	plt.legend(fontsize=15)
	plt.grid()

	plt.show()




