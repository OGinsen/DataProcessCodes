import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D,BatchNormalization,Activation

'''
ResNet network model
'''
class ResnetBlock(Model):

    def __init__(self,filters,strides=1,residual_path=False):
        super(ResnetBlock,self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv1D(filters,3,strides=strides,padding='same',use_bias=False,data_format='channels_last')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv1D(filters,3,strides=1,padding='same',use_bias=False,data_format='channels_last')
        self.b2 = BatchNormalization()

        # residual_ When path is true, the input is down sampled, that is, a 1x1 convolution kernel is used for convolution to ensure that X and f (x) dimensions are the same and can be added smoothly
        if residual_path:
            self.down_c1 = Conv1D(filters,1,strides=strides,padding='same',use_bias=False,data_format='channels_last')
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self,inputs):
        residual = inputs  #  Residual is equal to the input value itself, that is, residual=x
        # Calculate f (x) by convolution, BN layer and activation layer
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual) # The final output is the sum of two parts, that is, f (x) +x or F (x) +wx, and then activate the function
        return out

class ResNet18(Model):

    def __init__(self,block_list,initial_filters=64): # block_ list indicates that each block has several convolution layers
        super(ResNet18,self).__init__()
        self.num_blocks = len(block_list) # How many blocks are there
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv1D(self.out_filters,3,strides=1,padding='same',use_bias=False,data_format='channels_last')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # Build ResNeT network structure
        for block_id in range(len(block_list)): # The block_idth RESNET block
            for layer_id in range(block_list[block_id]): # The layer_idth convolutional layer

                if block_id != 0 and layer_id == 0: # Down sample the input of each block except the first block
                    block = ResnetBlock(self.out_filters,strides=2,residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters,residual_path=False)
                self.blocks.add(block) # Add the constructed block to ResNeT
            self.out_filters *= 2 # The convolution kernel number of the next block is twice that of the previous block
        self.p1 = tf.keras.layers.GlobalAveragePooling1D()
        self.f1 = tf.keras.layers.Dense(7,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2())

    def call(self,inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y