import os

'''
Make Training Set Label
'''

r_txt = open('TrainLabelRelax4ch.txt', 'w')
t_txt = open('TrainLabelTired4ch.txt', 'w')

for x, ys ,zs in os.walk('D:\\All_Datasets\\GestureClassificationDatasets\\tired', 'r'):
	for z in zs:
		if z[-3:]=='csv':
			emg_file = x + '\\' + z
			print(emg_file)
			label = z[17]
			t_txt.write(emg_file)
			t_txt.write(' ')
			t_txt.write(label)
			t_txt.write('\n')

t_txt.close()

for x, ys ,zs in os.walk('D:\\All_Datasets\\GestureClassificationDatasets\\relax', 'r'):
	for z in zs:
		if z[-3:]=='csv':
			emg_file = x + '\\' + z
			print(emg_file)
			label = z[17]
			r_txt.write(emg_file)
			r_txt.write(' ')
			r_txt.write(label)
			r_txt.write('\n')

r_txt.close()