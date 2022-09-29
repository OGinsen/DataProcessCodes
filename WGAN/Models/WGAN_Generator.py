import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Reshape,Conv1D,BatchNormalization,Activation,UpSampling1D,Flatten,Dense,Lambda,Concatenate,Input
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import plot_model

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

class Generator():

    def __init__(self,args,training = False):
        self.noise_dim = args['noise_dim']  # Original noise length
        self.channels = args['channels']    # One dimensional signal
        self.conv_activation = tf.nn.leaky_relu # Convolution activation function
        self.activation_function = args['activation_function']
        self.num_steps = args['num_steps']# Length of final generated signal
        self.seq_shape = (self.num_steps,self.channels)  # Shape of final generated signal
        self.sliding_window = args['sliding_window']  # The last layer of the generator is averaged with a sliding window (length 10)
        self.training_mode = training  # whether to print the model
        self.model = self.build_generator()

    def moving_avg(self,args):
        '''
        Implement the moving average filter of the last layer of the generator
        :param args:
        :return:
        '''
        input_ = args
        sliding_window = tf.signal.frame(
            signal=input_,
            frame_length=self.sliding_window,
            frame_step=1,
            pad_end=True,
            pad_value=0,
            axis=1,
            name='moving_average'
        )
        sliding_reshaped = tf.reshape(sliding_window,(-1,self.num_steps,self.sliding_window))
        mvg_avg = tf.reduce_mean(sliding_reshaped,axis=2,keepdims=True)
        return mvg_avg
    def Upsample(self, args):
        input_ = args
        input_Extended_dimension = input_[:, :, tf.newaxis, :]
        UpSampling = tf.image.resize(input_Extended_dimension, [input_Extended_dimension.shape[1] * 2, 1],
                                     method='bilinear')
        # print('UpSampling.shape:',UpSampling.shape)
        out = tf.reshape(UpSampling, (-1, UpSampling.shape[1], UpSampling.shape[3]))
        # print('out.shape:',out.shape)
        return out

    def build_generator(self,training=None):
        '''
        Construction builder model
        :return:
        '''

        input_ = Input(shape=(self.noise_dim, self.channels))
        cnn_1 = Conv1D(filters=128, kernel_size=4, padding='same', data_format='channels_last', name='CONV1')(input_)
        cnn_1 = BatchNormalization(momentum=0.8)(cnn_1,training=training)
        cnn_1 = Activation(self.conv_activation)(cnn_1)

        # print('cnn_1:', cnn_1.shape[0])
        # up_1 = UpSampling1D()(cnn_1)
        up_1 = Lambda(self.Upsample,name='UpSampling1')(cnn_1)
        # print(up_1)
        cnn_2 = Conv1D(filters=128, kernel_size=4, padding='same', name='CONV2')(up_1)
        cnn_2 = BatchNormalization(momentum=0.8)(cnn_2,training=training)
        cnn_2 = Activation(self.conv_activation)(cnn_2)

        # up_2 = UpSampling1D()(cnn_2)
        up_2 = Lambda(self.Upsample, name='UpSampling2')(cnn_2)
        cnn_3 = Conv1D(filters=64, kernel_size=4, padding='same', name='CONV3')(up_2)
        cnn_3 = BatchNormalization(momentum=0.8)(cnn_3,training=training)
        cnn_3 = Activation(self.conv_activation)(cnn_3)

        # up_3 = UpSampling1D()(cnn_3)
        up_3 = Lambda(self.Upsample, name='UpSampling3')(cnn_3)
        cnn_4 = Conv1D(filters=32, kernel_size=4, padding='same', name='CONV4')(up_3)
        cnn_4 = BatchNormalization(momentum=0.8)(cnn_4,training=training)
        cnn_4 = Activation(self.conv_activation)(cnn_4)

        cnn_5 = Conv1D(filters=16, kernel_size=4, padding='same', name='CONV5')(cnn_4)
        cnn_5 = BatchNormalization(momentum=0.8)(cnn_5,training=training)
        cnn_5 = Activation(self.conv_activation)(cnn_5)

        # cnn_6 = Conv1D(filters=8, kernel_size=4, padding='same', name='CONV6')(cnn_5)
        # cnn_6 = BatchNormalization(momentum=0.8)(cnn_6,training=training)
        # cnn_6 = Activation(self.conv_activation)(cnn_6)
        # # print('cnn_6.shape:', cnn_6.shape)
        #
        # cnn_7 = Conv1D(filters=4, kernel_size=4, strides=2, padding='same', name='CONV7')(cnn_6)
        # cnn_7 = BatchNormalization(momentum=0.8)(cnn_7,training=training)
        # cnn_7 = Activation(self.conv_activation)(cnn_7)
        #
        # cnn_8 = Conv1D(filters=2, kernel_size=4, strides=2, padding='same', name='CONV8')(cnn_7)
        # cnn_8 = BatchNormalization(momentum=0.8)(cnn_8,training=training)
        # cnn_8 = Activation(self.conv_activation)(cnn_8)
        #
        # cnn_9 = Conv1D(filters=self.channels, kernel_size=4, strides=2, padding='same', name='CONV9')(cnn_8)
        # cnn_9 = BatchNormalization(momentum=0.8)(cnn_9,training=training)
        # cnn_9 = Activation(self.conv_activation)(cnn_9)

        cnn_6 = Conv1D(filters=self.channels, kernel_size=4, strides=2, padding='same', name='CONV6')(cnn_5)
        cnn_6 = BatchNormalization(momentum=0.8)(cnn_6,training=training)
        cnn_6 = Activation(self.conv_activation)(cnn_6)

        flat = Flatten()(cnn_6)
        # flat = Flatten()(cnn_9)
        # print(flat.shape)
        # print('=====================')
        Den = Dense(self.num_steps * self.channels, name='Dense')(flat)
        Den = Activation(self.activation_function)(Den)
        # print('=====================================================')
        out = Reshape((self.num_steps, self.channels))(Den)
        # print('========================================================================')

        if self.sliding_window > 0:  # Determine whether the last layer uses a moving average
            out = Lambda(self.moving_avg, output_shape=self.seq_shape, name='mvg_avg')(out)
        model = Model(inputs=input_, outputs=out)

        # model = Sequential()
        # model.add(Reshape((self.noise_dim,self.channels),input_shape=(self.noise_dim,)))
        # model.add(Conv1D(filters=128,kernel_size=4,padding='same',data_format='channels_last',name='CONV1'))
        # model.add(BatchNormalization(momentum=0.8,training=self.training_mode))
        # model.add(Activation(self.conv_activation))
        #
        # model.add(UpSampling1D())
        # model.add(Conv1D(filters=128,kernel_size=4,padding='same',name='CONV2'))
        # model.add(BatchNormalization(momentum=0.8,training=self.training_mode))
        # model.add(Activation(self.conv_activation))
        #
        # model.add(UpSampling1D())
        # model.add(Conv1D(filters=64,kernel_size=4,padding='same',name='CONV3'))
        # model.add(BatchNormalization(momentum=0.8,training=self.training_mode))
        # model.add(Activation(self.conv_activation))
        #
        # model.add(UpSampling1D())
        # model.add(Conv1D(filters=32,kernel_size=4,padding='same',name='CONV4'))
        # model.add(BatchNormalization(momentum=0.8,training=self.training_mode))
        # model.add(Activation(self.conv_activation))
        #
        # model.add(Conv1D(filters=16,kernel_size=4,padding='same',name='CONV5'))
        # model.add(BatchNormalization(momentum=0.8,training=self.training_mode))
        # model.add(Activation(self.conv_activation))
        #
        # model.add(Conv1D(filters=self.channels,kernel_size=4,padding='same',name='CONV6'))
        # model.add(BatchNormalization(momentum=0.8,training=self.training_mode))
        # model.add(Activation(self.conv_activation))
        #
        # model.add(Flatten())
        # model.add(Dense(self.num_steps*self.channels,name='Dense'))
        # model.add(Activation(self.activation_function))
        # model.add(Reshape((self.num_steps,self.channels)))
        #
        # if self.sliding_window > 0 :#判断最后一层是否使用滑动平均
        #     model.add(Lambda(self.moving_avg,output_shape=self.seq_shape,name='mvg_avg'))

        # UNET
        # inputs = Input(shape=(self.noise_dim, self.channels))
        # # inputs = Reshape((self.noise_dim,self.channels),input_shape=(self.noise_dim,))(inputs)
        # # ---------------------------------#
        # #	获得五个有效特征层		      #
        # #	f1    8192,64				  #
        # #   f2    4096,128				  #
        # #   f3    2048,256				  #
        # #   f4    1024,512				  #
        # #   f5    512 ,512				  #
        # # ---------------------------------#
        # f1, f2, f3, f4, f5 = VGG16(inputs,training)
        # channels = [64, 128, 256, 512, 512]
        #
        # # 512,1024 -> 1024,512
        # P5_up = UpSampling1D(size=2)(f5)
        # # 1024,512 + 1024,512 -> 1024,1024
        # P4 = Concatenate(axis=2)([f4, P5_up])
        # # 1024,1024, -> 1024,512
        # P4 = Conv1D(channels[3], kernel_size=4, activation=self.conv_activation, padding='same',
        #             kernel_initializer=RandomNormal(stddev=0.02))(P4)
        # P4 = BatchNormalization(momentum=0.8)(P4,training)
        # P4 = Conv1D(channels[3], kernel_size=4, activation=self.conv_activation, padding='same',
        #             kernel_initializer=RandomNormal(stddev=0.02))(P4)
        # P4 = BatchNormalization(momentum=0.8)(P4,training)
        #
        # # 1024,512 -> 2048,512
        # P4_up = UpSampling1D(size=2)(P4)
        # # 2048,256 + 2048,512 -> 2048,768
        # P3 = Concatenate(axis=2)([f3, P4_up])
        # # 2048,768 -> 2048,256
        # P3 = Conv1D(channels[2], kernel_size=4, activation=self.conv_activation, padding='same',
        #             kernel_initializer=RandomNormal(stddev=0.02))(P3)
        # P3 = BatchNormalization(momentum=0.8)(P3,training)
        # P3 = Conv1D(channels[2], kernel_size=4, activation=self.conv_activation, padding='same',
        #             kernel_initializer=RandomNormal(stddev=0.02))(P3)
        # P3 = BatchNormalization(momentum=0.8)(P3,training)
        #
        # # 2048,256 -> 4096,256
        # P3_up = UpSampling1D(size=2)(P3)
        # # 4096,128 + 4096,256 -> 4096,384
        # P2 = Concatenate(axis=2)([f2, P3_up])
        # # 4096,384 -> 4096,128
        # P2 = Conv1D(channels[1], kernel_size=4, activation=self.conv_activation, padding='same',
        #             kernel_initializer=RandomNormal(stddev=0.02))(P2)
        # P2 = BatchNormalization(momentum=0.8)(P2,training)
        # P2 = Conv1D(channels[1], kernel_size=4, activation=self.conv_activation, padding='same',
        #             kernel_initializer=RandomNormal(stddev=0.02))(P2)
        # P2 = BatchNormalization(momentum=0.8)(P2,training)
        #
        # # 4096,128 -> 8192,128
        # P2_up = UpSampling1D(size=2)(P2)
        # # 8192,64 + 8192,128 -> 8192,192
        # P1 = Concatenate(axis=2)([f1, P2_up])
        # # 8192,192 -> 8192,64
        # P1 = Conv1D(channels[0], kernel_size=4, activation=self.conv_activation, padding='same',
        #             kernel_initializer=RandomNormal(stddev=0.02))(P1)
        # P1 = BatchNormalization(momentum=0.8)(P1,training)
        # P1 = Conv1D(channels[0], kernel_size=4, activation=self.conv_activation, padding='same',
        #             kernel_initializer=RandomNormal(stddev=0.02))(P1)
        # P1 = BatchNormalization(momentum=0.8)(P1,training)
        #
        # # 8192,64 -> 8192,1
        # P1 = Conv1D(1, kernel_size=1, activation=self.conv_activation)(P1)
        # P1 = BatchNormalization(momentum=0.8)(P1,training)
        #
        # P1 = Flatten()(P1)
        # print('----------------P1.shape:', P1.shape)
        # P1 = Dense(self.num_steps, activation='tanh')(P1)
        # print('----------------P1.shape:', P1.shape)
        # P1 = Reshape(self.seq_shape)(P1)
        #
        # if self.sliding_window > 0:  # 10 > 0
        #     P1 = Lambda(self.moving_avg, output_shape=self.seq_shape, name='mvg_avg')(P1)  # (2000,1)
        #
        # model = Model(inputs=inputs, outputs=P1)
        if self.training_mode:# Determine whether to print the model
            print('Generator model:')
            model.summary()
            model_json = model.to_json()

            with open('./Output/Generator.json','w') as json_file:
                json_file.write(model_json)

            file_name = './Output/Generator.png'
            plot_model(model,to_file=file_name,show_shapes=True)
        return model

    def save(self,index=-1,G = -1,ch=-1):
        # Store generator model parameters
        if index == -1:
            file_path = './SavedModels/G'+str(G)+'/ch'+str(ch)+'/Generator.h5'
        else:
            file_path = './SavedModels/G'+str(G)+'/ch'+str(ch)+'/Generator_' + str(index) + '.h5'
        self.model.save_weights(file_path)

    def load(self,index=-1,G = -1,ch=-1):
        # Load model and model parameters
        if index == -1:
            file_path = './SavedModels/G'+str(G)+'/ch'+str(ch)+'/Generator.h5'
        else:
            file_path = './SavedModels/G'+str(G)+'/ch'+str(ch)+'/Generator_' + str(index) + '.h5'
        self.model = self.build_generator()
        self.model.load_weights(file_path)


    def predict(self,args):
        return self.model.predict(args)
