import os, random, shutil


'''
Make a validation set
'''

def moveFile(fileDir,tarDir):
	pathDir = os.listdir(fileDir)  # Get the original path of the file
	G0,G1,G2,G3,G4,G5,G6 = [],[],[],[],[],[],[]
	for G in pathDir:
		if G[17] == '0':
			G0.append(G)
		elif G[17] == '1':
			G1.append(G)
		elif G[17] == '2':
			G2.append(G)
		elif G[17] == '3':
			G3.append(G)
		elif G[17] == '4':
			G4.append(G)
		elif G[17] == '5':
			G5.append(G)
		elif G[17] == '6':
			G6.append(G)
	All_G = [G0,G1,G2,G3,G4,G5,G6]
	for i in range(len(All_G)):
		filenumber = len(All_G[i])
		rate = 0.1 # Customize the proportion of extracted files. For example, 10 out of 100 files is 0.1
		picknumber = int(filenumber * rate)  # Take a certain number of files from the folder according to the rate ratio
		sample = random.sample(All_G[i], picknumber)  # Select a certain number of sample files at random
		print(sample)
		for name in sample:
			shutil.move(fileDir + name, tarDir + name)



if __name__ == '__main__':
	relax_fileDir = "D:\\All_Datasets\\GestureClassificationDatasets\\relax\\"  # Source folder path
	relax_tarDir = 'D:\\All_Datasets\\CNNValidationSet\\relax\\'  # Move to new folder path

	tired_fileDir = "D:\\All_Datasets\\GestureClassificationDatasets\\tired\\"  # Source folder path
	tired_tarDir = 'D:\\All_Datasets\\CNNValidationSet\\tired\\'  # Move to new folder path

	moveFile(relax_fileDir,relax_tarDir)
	moveFile(tired_fileDir,tired_tarDir)