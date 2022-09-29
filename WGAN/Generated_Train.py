
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from Classification.DeepLearning.CNN_Models import ResNet18
from tensorflow.keras.losses import SparseCategoricalCrossentropy
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
np.set_printoptions(threshold=np.inf)

# train_path = './Training_set'
# train_txt = './Train_label_mixed.txt'
# x_train_savepath = './Datasets_npy/mixed/x_train.npy'
# y_train_savepath = './Datasets_npy/mixed/y_train.npy'
#############################################################
# train_txt = 'TrainLabelRelax4ch.txt'
# x_train_savepath = './Datasets_npy/relax/x_train_4ch.npy'
# y_train_savepath = './Datasets_npy/relax/y_train_4ch.npy'
train_txt = 'GeneratedSignals_npy/GeneratedSignals.txt'
x_train_savepath = 'GeneratedSignals_npy/x_train_ch0.npy'
y_train_savepath = 'GeneratedSignals_npy/y_train_ch0.npy'
def generated(txt):
	'''
	Used to generate training data sets
	:param txt: Label file
	:return: Training sets, training labels
	'''
	file = open(txt,'r') # Open txt file as read-only
	contents = file.readlines() # Read all lines in the file
	file.close() # Close file
	x , y_ = [] , [] # Create an empty list
	for content in contents: # Row by row extraction
		value = content.split() # Split with spaces
		datasets_path = value[0] # Concatenate file paths and file names
		datasets = np.loadtxt(datasets_path, delimiter=',')[512:512+1024]
		print(datasets.shape)

		x.append(datasets)
		y_.append(value[1])
		print('loading:' + content) # Print status prompt
	x = np.array(x)
	print('=================================',x.shape)
	y_ = np.array(y_)
	y_ = y_.astype(np.int64) # Change to 64 bit integer
	print('x.shape, y_.shape:',x.shape, y_.shape)
	return x,y_

if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath):
	print('-------------Load Datasets--------------')
	x_train_save = np.load(x_train_savepath)
	# x_train_save = np.load(x_train_savepath)[:,:,2:3] # Training channel 2
	y_train = np.load(y_train_savepath)
	print('x_train_save.shape:',x_train_save.shape)
	print('y_train_save.shape:',y_train.shape)
	x_train = x_train_save.reshape(x_train_save.shape[0],x_train_save.shape[1],1)

	# Randomly scramble data
	x_train_size = x_train.shape[0]
	arr = np.arange(x_train_size)
	np.random.shuffle(arr)
	x_train = x_train[arr]
	y_train = y_train[arr]

	# Divide training set and test set
	rate = 0.1 # Take 10% of the training set as the test set
	x_train = x_train[:-int(rate*x_train_size)]
	y_train = y_train[:-int(rate*x_train_size)]
	print('x_train.shape:', x_train.shape)
	print('y_train.shape:', y_train.shape)
	x_test = x_train[-int(rate*x_train_size):]
	y_test = y_train[-int(rate*x_train_size):]
	print('x_test.shape:', x_test.shape)
	print('y_test.shape:', y_test.shape)

else:
	print('--------------Generate Datasets---------------')
	x_train , y_train = generated(train_txt)

	print('--------------Save Datasets------------')

	print('x_train.shape,y_train.shape:',x_train.shape,y_train.shape)
	np.save(x_train_savepath,x_train)
	np.save(y_train_savepath,y_train)

model = ResNet18([2,2,2,2])

model.compile(optimizer=Adam(learning_rate=0.001),
			  	loss=SparseCategoricalCrossentropy(from_logits=False),
			  	metrics=['sparse_categorical_accuracy'])

# Read model
# checkpoint_save_path = './SavedModels/relax/4ch/checkpoint_ResNet18/ResNet18_4ch.ckpt'
checkpoint_save_path = 'GeneratedSignalsSavedModels/ch0/checkpoint_ResNet18/ResNet18_4ch.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
	print('-------------load the model-----------------')
	model.load_weights(checkpoint_save_path)

# Save model
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
												 save_weights_only=True,
												 save_best_only=True)
history = model.fit(x_train,y_train,batch_size=128,epochs=500,validation_data=(x_test,y_test),validation_freq=1,callbacks=[cp_callback])

model.summary()


# file = open('./Output/relax/4ch/weights_ResNet18_4ch.txt','w')
file = open('GeneratedSignalsOutput/ch0/weights_ResNet18_ch0.txt','w')
for v in model.trainable_variables:
	file.write(str(v.name) + '\n')
	file.write(str(v.shape) + '\n')
	file.write(str(v.numpy()) + '\n')
file.close()

##############################################   show  #######################################

# Display ACC and loss curves of training set and verification set
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1,2,1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
# plt.show()
# plt.savefig('./Output/relax/4ch/ResNet18_4ch.png')
plt.savefig('GeneratedSignalsOutput/ch0/ResNet18_ch0.png')