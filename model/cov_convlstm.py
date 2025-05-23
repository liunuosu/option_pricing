import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Model
from keras.layers import ConvLSTM2D, LSTM, BatchNormalization, \
                            Input, Dense, Reshape, Concatenate, Masking
from keras.utils import plot_model
from utils.loss import masked_mse

class CovConvLSTM:

    def __init__(self, x_iv_train, x_cov_train, y_iv_train, \
                 x_iv_val=None, x_cov_val=None, y_iv_val=None, config=None):
        self.read_config(config) # Read the parameters and set the data
        self.x_iv_train = x_iv_train
        self.x_cov_train = x_cov_train
        self.target_train = y_iv_train
        self.x_iv_val = x_iv_val
        self.x_cov_val = x_cov_val
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
        # height = len(data_train['moneyness'].unique())
        # width = len(data_train['maturity'].unique())
        num_covariates = len(self.covariate_columns)

        # skip seed for now, first check results
        
        iv_input = Input(shape=(time_steps, height, width, 1), name="iv_input")
        x_iv = Masking(mask_value=0.0)(iv_input)

        for i in range(self.num_layer-1):
            x_iv = ConvLSTM2D(filters=self.filters, 
                                    kernel_size=(self.kernel_height, self.kernel_width),
                                    strides=(self.strides_dim, self.strides_dim),
                                    padding=self.padding, 
                                    return_sequences=True,
                                    kernel_initializer=self.kernel_initializer,
                                    recurrent_initializer=self.recurrent_initializer,
                                    activation=self.conv_activation,
                                    recurrent_activation=self.recurrent_activation)(iv_input)
            x_iv = BatchNormalization()(x_iv)

        x_iv = ConvLSTM2D(filters=self.filters, 
                                    kernel_size=(self.kernel_height, self.kernel_width),
                                    strides=(self.strides_dim, self.strides_dim),
                                    padding=self.padding, 
                                    return_sequences=False,
                                    kernel_initializer=self.kernel_initializer,
                                    recurrent_initializer=self.recurrent_initializer,
                                    activation=self.conv_activation,
                                    recurrent_activation=self.recurrent_activation)(x_iv)
        x_iv = BatchNormalization()(x_iv)

        cov_input = Input(shape=(time_steps, num_covariates), name="cov_input")
        # LSTM layer for the covariates
        x_cov = LSTM(units=64, return_sequences=False)(cov_input)   # (batch_size, 64)
        x_cov = Dense(units=height * width, activation='relu')(x_cov)  
        x_cov = Reshape((height, width, 1))(x_cov)               # (batch_size, H, W, 1)

        x = Concatenate(axis=-1)([x_iv, x_cov])  # Combine along channel axis -> (H, W, 65)
        x = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same')(x)

        self.model = Model(inputs=[iv_input, cov_input], outputs=x)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.model.compile(loss=masked_mse, optimizer=optimizer)
        print(self.model.summary())

    def fit(self):
        self.history = self.model.fit([self.x_iv_train, self.x_cov_train], self.target_train,
                validation_data=([self.x_iv_val, self.x_cov_val], self.target_val),
                epochs=self.epochs, batch_size=self.batch_size, shuffle=False,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=self.patience,
                                                            mode='min')])
        best_epoch = int(np.argmin(self.history.history['val_loss'])) + 1
        best_val_loss = self.history.history['val_loss'][best_epoch]
        train_loss = self.history.history['loss']
        val_loss = self.history.history.get('val_loss')
        return best_epoch, best_val_loss, train_loss, val_loss
    
    def fit_test(self, num_epoch):
        self.model.fit([self.x_iv_train, self.x_cov_train], self.target_train,
                epochs=num_epoch, batch_size=self.batch_size, shuffle=False)
    
    def pred(self, x_iv, x_cov): 
        pred = self.model.predict([x_iv, x_cov])
        return pred
    
    def plot_architecture(self, filename='covconvlstm.png'):
        plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=False,
                   dpi=300, rankdir='TB')
    