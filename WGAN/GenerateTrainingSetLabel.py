import os

gen_txt = open('GeneratedSignals_npy/GeneratedSignals.txt', 'w')

for x, ys ,zs in os.walk('GeneratedSignals', 'r'):
	for z in zs:
		if z[-3:]=='csv':
			emg_file = x + '\\' + z
			print(emg_file)
			label = z[14]
			gen_txt.write(emg_file)
			gen_txt.write(' ')
			gen_txt.write(label)
			gen_txt.write('\n')

gen_txt.close()

