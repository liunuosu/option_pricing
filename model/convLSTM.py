import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import ConvLSTM2D, BatchNormalization
from keras.layers import Masking, InputLayer
from keras.utils import plot_model
from utils.loss import masked_mse
import numpy as np

class ConvLSTM:

    def __init__(self, x_iv_train, y_iv_train, \
                 x_iv_val=None, y_iv_val=None, config=None):
        self.read_config(config) # Read the parameters and set the data
        self.x_iv_train = x_iv_train
        self.target_train = y_iv_train
        self.x_iv_val = x_iv_val
        self.target_val = y_iv_val

    def read_config(self, config):

        self.patience = config['training']['patience']
        self.epsilon = config['training']['epsilon']
        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs']
        self.seed = config['training']['seed']
        self.learning_rate = config['training']['lr']

        self.window_size = config['data']['window_size']
        self.run = config['data']['run']
        self.covariate_columns = config['data']['covariates']
        self.option_type = config['data']['option']
        self.smooth = config['data']['smooth']
        self.h_step = config['data']['h_step']

        self.filters = config['model']['filters'] # 16 32 64 128 filter within the conv2DLSTM layer
        self.kernel_height = config['model']['kernel_height']
        self.kernel_width = config['model']['kernel_width'] # 1 to 5 (maturity) mxn -> 9x5
        self.num_layer = config['model']['num_layer'] # Any positive integer >0
        self.strides_dim = config['model']['strides_dim'] #: !!int 1 # assumes strides to be same across the two dimensions 
        self.kernel_initializer = config['model']['kernel_initializer'] 
        self.recurrent_initializer = config['model']['recurrent_initializer']
        self.padding = config['model']['padding']
        self.conv_activation = config['model']['conv_activation']
        self.recurrent_activation = config['model']['recurrent_activation']

    def compile(self):
        # set seed before compiling
        tf.random.set_seed(self.seed)

        time_steps = self.window_size
        _, window, height, width, _ = self.x_iv_train.shape
        channels = 1 
        # height = len(data_train['moneyness'].unique())
        # width = len(data_train['maturity'].unique())
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(time_steps, height, width, channels)))
        self.model.add(Masking(mask_value=0.0))
        # self.model.add(Masking(mask_value=0.0))
        # ConvLSTM2D expects 5D input: (batch, time, height, width, channels)

        for i in range(self.num_layer-1):
            self.model.add(ConvLSTM2D(filters=self.filters, 
                                      kernel_size=(self.kernel_height, self.kernel_width),
                                      strides=(self.strides_dim, self.strides_dim),
                                    padding=self.padding, 
                                    return_sequences=True,
                                    kernel_initializer=self.kernel_initializer,
                                    recurrent_initializer=self.recurrent_initializer,
                                    activation=self.conv_activation,
                                    recurrent_activation=self.recurrent_activation
                                ))
            self.model.add(BatchNormalization())

        self.model.add(ConvLSTM2D(filters=self.filters, 
                                  kernel_size=(self.kernel_height, self.kernel_width),
                                  strides=(self.strides_dim, self.strides_dim),
                                  padding=self.padding, 
                                  return_sequences=False,
                                  kernel_initializer=self.kernel_initializer,
                                  recurrent_initializer=self.recurrent_initializer,
                                  activation=self.conv_activation,
                                  recurrent_activation=self.recurrent_activation))
        self.model.add(BatchNormalization())

        # Final 3D convolution to map to the next frame
        self.model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1),
                                        activation='sigmoid', padding='same'))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.model.compile(loss=masked_mse, optimizer=self.optimizer)

        # Double check the architecture, and the activaiton function
        print(self.model.summary())


    def fit(self):
        self.history = self.model.fit(self.x_iv_train, self.target_train,
                validation_data=(self.x_iv_val, self.target_val),
                epochs=self.epochs, batch_size=self.batch_size, shuffle=False,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=self.patience,
                                                            mode='min')])
        best_epoch = int(np.argmin(self.history.history['val_loss'])) + 1
        best_val_loss = self.history.history['val_loss'][best_epoch-1]
        train_loss = self.history.history['loss']
        val_loss = self.history.history.get('val_loss')
        return best_epoch, best_val_loss, train_loss, val_loss
    
    def fit_test(self, num_epoch):
        self.model.fit(self.x_iv_train, self.target_train,
                epochs=num_epoch, batch_size=self.batch_size, shuffle=False)
    
    def pred(self, x_iv): 
        pred = self.model.predict(x_iv)
        return pred
    
    def plot_architecture(self, filename='convlstm.png'):
        plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=False,
                   dpi=300, rankdir='TB')
    