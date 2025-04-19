import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import ConvLSTM2D, LSTM, BatchNormalization, Conv3D, Conv2D,Input, Dense, Reshape, Concatenate
from keras.layers import Masking
import yaml
from utils.loss import masked_mse
from utils.metrics import calculate_ivrmse_mask, calculate_r_oos_mask
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

        self.learning_rate = config['model']['lr']

        self.window_size = config['data']['window_size']
        self.run = config['data']['run']
        self.covariate_columns = config['data']['covariates']
        self.option_type = config['data']['option']
        self.smooth = config['data']['smooth']
        self.h_step = config['data']['h_step']

    def compile(self):

        time_steps = self.window_size
        _, window, height, width, _ = self.x_iv_train.shape
        channels = 1 
        # height = len(data_train['moneyness'].unique())
        # width = len(data_train['maturity'].unique())
        model = Sequential()

        # ConvLSTM2D expects 5D input: (batch, time, height, width, channels)
        model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                            padding='same', return_sequences=True,
                            input_shape=(time_steps, height, width, channels)))
        model.add(BatchNormalization())

        model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                            padding='same', return_sequences=False))
        model.add(BatchNormalization())

        # Final 3D convolution to map to the next frame
        model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1),
                                        activation='sigmoid', padding='same'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=self.epsilon)
        model.compile(loss=masked_mse, optimizer=optimizer)

        # Double check the architecture, and the activaiton function
        print(model.summary())


    def fit(self):
        self.history = self.model.fit(self.x_iv_train, self.target_train,
                validation_data=(self.x_iv_val, self.target_val),
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
        self.model.fit(self.x_iv_train, self.target_train,
                epochs=num_epoch, batch_size=self.batch_size, shuffle=False)
    
    def pred(self, x_iv): 
        pred = self.model.predict(x_iv)
        return pred
    