import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import InputSpec,Layer,Conv1D,Input,Flatten,LeakyReLU,Dropout,BatchNormalization,Lambda,Reshape,Dense,Concatenate
import pywt
from tensorflow import keras
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
class MinibatchDiscrimination(Layer):
    """Concatenates to each sample information about how different the input
    features for that sample are from features of other samples in the same
    minibatch, as described in Salimans et. al. (2016). Useful for preventing
    GANs from collapsing to a single output. When using this layer, generated
    samples and reference samples should be in separate batches.

    # Example

    ```python
        # apply a convolution 1d of length 3 to a sequence with 10 timesteps,
        # with 64 output filters
        model = Sequential()
        model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
        # now model.output_shape == (None, 10, 64)

        # flatten the output so it can be fed into a minibatch discrimination layer
        model.add(Flatten())
        # now model.output_shape == (None, 640)

        # add the minibatch discrimination layer
        model.add(MinibatchDiscrimination(5, 3))
        # now model.output_shape = (None, 645)
    ```

    # Arguments
        nb_kernels: Number of discrimination kernels to use
            (dimensionality concatenated to output).
        kernel_dim: The dimensionality of the space where closeness of samples
            is calculated.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        weights: list of numpy arrays to set as initial weights.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        input_dim: Number of channels/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.

    # Input shape
        2D tensor with shape: `(samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(samples, input_dim + nb_kernels)`.

    # References
        - [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
    """

    def __init__(self, nb_kernels, kernel_dim, init='glorot_uniform', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        self.init = initializers.get(init)
        self.nb_kernels = nb_kernels
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2

        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight(shape=(self.nb_kernels, input_dim, self.kernel_dim),
            initializer=self.init,
            name='kernel',
            regularizer=self.W_regularizer,
            trainable=True,
            constraint=self.W_constraint)

        # Set built to true.
        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, mask=None):
        activation = K.reshape(K.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
        diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), axis=2)
        minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
        return K.concatenate([x, minibatch_features], 1)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], input_shape[1]+self.nb_kernels

    def get_config(self):
        config = {'nb_kernels': self.nb_kernels,
                  'kernel_dim': self.kernel_dim,
                  'init': self.init.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MinibatchDiscrimination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Discriminator():

    def __init__(self, args, training=False):

        self.channels = args['channels']   # One dimensional signal channel
        self.num_steps = args['num_steps'] # Length of final generated signal
        self.seq_shape = (self.num_steps,self.channels) # Shape of final generated signal
        self.training_mode = training # whether to print the model

        self.dropout_rate = args['dropout_rate']

        # Determining the parameters of wavelet decomposition
        self.wavelet_mother = args['wavelet_mother'] # Wavelet generating function
        self.wavelet_levels = args['wavelet_levels'] # 2-level discrete wavelet transform
        self.wavelet_trainable = args['wavelet_trainable']
        self.use_mini_batch = args['use_mini_batch']

        self.moving_avg_window = args['moving_avg_window'] # Sliding window size for envelope signal
        self.model = self.build_discriminator()# Construct discriminator model

    def make_wavelet_expansion(self, input_tensor):
        # Perform wavelet transform
        low_pass, high_pass = pywt.Wavelet(self.wavelet_mother).filter_bank[:2]
        low_pass_filter = np.array(low_pass)
        high_pass_filter = np.array(high_pass)
        n_levels = self.wavelet_levels
        trainable = self.wavelet_trainable

        wv_kwargs = {
            "filters": 1,
            "kernel_size": len(low_pass),
            "strides": 2,
            "use_bias": False,
            "padding": "same",
            "trainable": trainable,
        }

        approximation_coefficients = []
        detail_coefficients = []

        last_approximant = input_tensor
        for i in range(n_levels):
            lpf = low_pass_filter
            hpf = high_pass_filter
            a_n = Conv1D(
                kernel_initializer=keras.initializers.Constant(lpf.reshape((-1, 1))),
                name="low_pass_{}".format(i),
                **wv_kwargs
            )(last_approximant)
            d_n = Conv1D(
                kernel_initializer=keras.initializers.Constant(hpf.reshape((-1, 1))),
                name="high_pass_{}".format(i),
                **wv_kwargs,
            )(last_approximant)

            detail_coefficients.append(d_n)
            approximation_coefficients.append(a_n)
            last_approximant = a_n

        return approximation_coefficients, detail_coefficients

    def envelops(self,args):
        # Envelope signal
        input_ = args
        abs_envelope = K.abs(input_)
        envelope = tf.signal.frame(
            abs_envelope,
            frame_length=self.moving_avg_window,
            frame_step=1,
            pad_end=True,
            pad_value=0,
            axis=1,
            name='envelope_moving_average'
        )
        envelope_reshaped = K.reshape(envelope, (-1, self.num_steps, self.moving_avg_window))
        envelope_mean = K.mean(envelope_reshaped, axis=2, keepdims=True)
        return envelope_mean

    def build_discriminator(self,training=None):

        input_ = Input(shape=self.seq_shape)


        flat = Flatten()(input_)
        #CNN on raw signal
        cnn_1 = Conv1D(filters=16,kernel_size=3,strides=2,padding='same',name='raw_conv_1')(input_)
        cnn_1 = LeakyReLU(alpha=0.2)(cnn_1)
        cnn_1 = Dropout(rate=self.dropout_rate)(cnn_1)

        cnn_2 = Conv1D(filters=32,kernel_size=3,strides=2,padding='same',name='raw_conv_2')(cnn_1)
        cnn_2 = BatchNormalization(momentum=0.8)(cnn_2,training=training)
        cnn_2 = LeakyReLU(alpha=0.2)(cnn_2)
        cnn_2 = Dropout(self.dropout_rate)(cnn_2)

        cnn_3 = Conv1D(filters=64,kernel_size=3,strides=2,padding='same',name='raw_conv_3')(cnn_2)
        cnn_3 = BatchNormalization(momentum=0.8)(cnn_3,training=training)
        cnn_3 = LeakyReLU(alpha=0.2)(cnn_3)
        cnn_3 = Dropout(rate=self.dropout_rate)(cnn_3)

        cnn_4 = Conv1D(filters=32,kernel_size=3,strides=2,padding='same',name='raw_conv_4')(cnn_3)
        cnn_4 = BatchNormalization(momentum=0.8)(cnn_4,training=training)
        cnn_4 = LeakyReLU(alpha=0.2)(cnn_4)
        cnn_4 = Dropout(self.dropout_rate)(cnn_4)

        cnn_4_out = Flatten()(cnn_4)


        #CNN on FFT of raw signal
        fft = Lambda(tf.signal.rfft)(flat)# Fourier transform
        fft_abs = Lambda(K.abs)(fft)
        fft_abs = Reshape(target_shape=(fft_abs.shape[1],1),name='fft_abs')(fft_abs)
        fft_cnn_1 = Conv1D(filters=16,kernel_size=3,strides=2,padding='same',name='fft_conv_1')(fft_abs)
        fft_cnn_1 = LeakyReLU(alpha=0.2)(fft_cnn_1)
        fft_cnn_1 = Dropout(rate=self.dropout_rate)(fft_cnn_1)
        fft_cnn_2 = Conv1D(filters=32,kernel_size=3,strides=2,padding='same',name='fft_conv_2')(fft_cnn_1)
        fft_cnn_2 = BatchNormalization(momentum=0.8)(fft_cnn_2,training=training)
        fft_cnn_2 = LeakyReLU(alpha=0.2)(fft_cnn_2)
        fft_cnn_2 = Dropout(rate=self.dropout_rate)(fft_cnn_2,training=training)

        fft_cnn_3 = Conv1D(filters=64,kernel_size=3,strides=2,padding='same',name='fft_conv_3')(fft_cnn_2)
        fft_cnn_3 = BatchNormalization(momentum=0.8)(fft_cnn_3,training=training)
        fft_cnn_3 = LeakyReLU(alpha=0.2)(fft_cnn_3)
        fft_cnn_3 = Dropout(rate=self.dropout_rate)(fft_cnn_3)

        fft_cnn_4 = Conv1D(filters=64,kernel_size=3,strides=2,padding='same',name='fft_conv_4')(fft_cnn_3)
        fft_cnn_4 = BatchNormalization(momentum=0.8)(fft_cnn_4,training=training)
        fft_cnn_4 = LeakyReLU(alpha=0.2)(fft_cnn_4)
        fft_cnn_4 = Dropout(rate=self.dropout_rate)(fft_cnn_4)

        fft_cnn_4_out = Flatten()(fft_cnn_4)

        # CNN on envelope
        envelope_window = Lambda(function=self.envelops,output_shape=self.seq_shape,name='envelope')(input_)
        envelope_window  = Flatten()(envelope_window )
        print('=====================')
        print(envelope_window.shape)
        envelope_fft = Lambda(tf.signal.rfft)(envelope_window)
        envelope_fft_abs = Lambda(K.abs)(envelope_fft)
        print('=============================')
        print(envelope_fft_abs.shape)
        envelope_fft_abs = Reshape((envelope_fft_abs.shape[1], 1))(envelope_fft_abs)

        envelope_cnn_1 = Conv1D(filters=16,kernel_size=3,strides=2,padding='same',name='env_conv_1')(envelope_fft_abs)
        envelope_cnn_1 = LeakyReLU(alpha=0.2)(envelope_cnn_1)
        envelope_cnn_1 = Dropout(rate=self.dropout_rate)(envelope_cnn_1)

        envelope_cnn_2 = Conv1D(filters=32,kernel_size=3,strides=2,padding='same',name='env_conv_2')(envelope_cnn_1)
        envelope_cnn_2 = BatchNormalization(momentum=0.8)(envelope_cnn_2,training=training)
        envelope_cnn_2 = LeakyReLU(alpha=0.2)(envelope_cnn_2)
        envelope_cnn_2 = Dropout(rate=self.dropout_rate)(envelope_cnn_2)

        envelope_cnn_3 = Conv1D(filters=64,kernel_size=3,strides=2,padding='same',name='env_conv_3')(envelope_cnn_2)
        envelope_cnn_3 = BatchNormalization(momentum=0.8)(envelope_cnn_3,training=training)
        envelope_cnn_3 = LeakyReLU(alpha=0.2)(envelope_cnn_3)
        envelope_cnn_3 = Dropout(rate=self.dropout_rate)(envelope_cnn_3)

        envelope_cnn_4 = Conv1D(filters=32,kernel_size=3,strides=2,padding='same',name='env_conv_4')(envelope_cnn_3)
        envelope_cnn_4 = BatchNormalization(momentum=0.8)(envelope_cnn_4)
        envelope_cnn_4 = LeakyReLU(alpha=0.2)(envelope_cnn_4)
        envelope_cnn_4 = Dropout(rate=self.dropout_rate)(envelope_cnn_4)

        envelope_cnn_4_out = Flatten()(envelope_cnn_4)

        # Wavelet Expansion
        approximation_coefficients, detail_coefficients = self.make_wavelet_expansion(input_)
        features_list = []
        features_list.extend(detail_coefficients)
        features_list.append(approximation_coefficients[-1])
        wavelet_concatenate = Concatenate(axis=1)(features_list)

        wavelet_cnn_1 = Conv1D(filters=16,kernel_size=3,strides=2,padding='same',name='wavelet_conv_1')(wavelet_concatenate)
        wavelet_cnn_1 = LeakyReLU(alpha=0.2)(wavelet_cnn_1)
        wavelet_cnn_1 = Dropout(rate=self.dropout_rate)(wavelet_cnn_1)

        wavelet_cnn_2 = Conv1D(filters=32,kernel_size=3,strides=2,padding='same',name='wavelet_conv_2')(wavelet_cnn_1)
        wavelet_cnn_2 = BatchNormalization(momentum=0.8)(wavelet_cnn_2,training=training)
        wavelet_cnn_2 = LeakyReLU(alpha=0.2)(wavelet_cnn_2)
        wavelet_cnn_2 = Dropout(rate=self.dropout_rate)(wavelet_cnn_2)

        wavelet_cnn_3 = Conv1D(filters=64,kernel_size=3,strides=2,padding='same',name='wavelet_conv_3')(wavelet_cnn_2)
        wavelet_cnn_3 = BatchNormalization(momentum=0.8)(wavelet_cnn_3,training=training)
        wavelet_cnn_3 = LeakyReLU(alpha=0.2)(wavelet_cnn_3)
        wavelet_cnn_3 = Dropout(rate=self.dropout_rate)(wavelet_cnn_3)

        wavelet_cnn_4 = Conv1D(filters=32,kernel_size=3,strides=2,padding='same',name='wavelet_conv_4')(wavelet_cnn_3)
        wavelet_cnn_4 = BatchNormalization(momentum=0.8)(wavelet_cnn_4,training=training)
        wavelet_cnn_4 = LeakyReLU(alpha=0.2)(wavelet_cnn_4)
        wavelet_cnn_4 = Dropout(rate=self.dropout_rate)(wavelet_cnn_4)

        wavelet_cnn_4_out = Flatten()(wavelet_cnn_4)

        #Mini batch discrimination
        mini_disc = MinibatchDiscrimination(30, 10)(flat)

        if self.use_mini_batch:
            concatenate = Concatenate()([cnn_4_out,fft_cnn_4_out,envelope_cnn_4_out,wavelet_cnn_4_out,mini_disc])
            # concatenate = Concatenate()([cnn_4_out, fft_cnn_4_out, mini_disc])
        else:
            concatenate = Concatenate()([cnn_4_out,fft_cnn_4_out,envelope_cnn_4_out,wavelet_cnn_4_out])
            # concatenate = Concatenate()([cnn_4_out, fft_cnn_4_out, wavelet_cnn_4_out])

        # mlp = Dense(1,activation='sigmoid')(concatenate)
        mlp = Dense(1)(concatenate) # The last layer does not perform sigmoid first

        model = Model(input_,mlp)  # Whole discriminator model

        if self.training_mode:# Determine whether to print the model
            print('Critic model:')
            model.summary()

            file_name = './Output/Discriminator.png'
            plot_model(model,to_file=file_name,show_shapes=True)
        return model

    def save(self,index=-1,G = -1,ch=-1):
        # Store discriminator model parameters
        if index == -1:
            file_path = './SavedModels/G'+str(G)+'/ch'+str(ch)+'/Discriminator.h5' # This is the final parameter after saving the training
        else:
            file_path = './SavedModels/G' + str(G) + '/ch' + str(ch) + '/Discriminator_' + str(index) + '.h5' # This is the parameter to save the training interruption
        self.model.save_weights(filepath=file_path)

    def load(self,index=-1,G = -1,ch=-1):
        # Load generator model and its parameters
        if index==-1:
            file_path = './SavedModels/G'+str(G)+'/ch'+str(ch)+'/Discriminator.h5'
        else:
            file_path = './SavedModels/G'+str(G)+'/ch'+str(ch)+'/Discriminator_' + str(index) + '.h5'
        self.model = self.build_discriminator()
        self.model.load_weights(file_path)

    def predict(self,args):
        return self.model.predict(args)



