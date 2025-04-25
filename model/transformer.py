import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import ConvLSTM2D, LSTM, BatchNormalization, Conv3D, Conv2D,Input, Dense, Reshape, Concatenate
from keras.layers import Masking, InputLayer
import yaml
from utils.loss import masked_mse
import numpy as np

from keras.layers import LayerNormalization, MultiHeadAttention, Dense, Dropout, Layer, Input, TimeDistributed, Reshape, Flatten
from keras.models import Model

class Transformer:

    def __init__(self, x_iv_train, y_iv_train, \
                 x_iv_val=None, y_iv_val=None, config=None):
        self.read_config(config) 
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

        # self.filters = config['model']['filters'] # 16 32 64 128 filter within the conv2DLSTM layer
        # self.kernel_height = config['model']['kernel_height']
        # self.kernel_width = config['model']['kernel_width'] # 1 to 5 (maturity) mxn -> 9x5
        # self.num_layer = config['model']['num_layer'] # Any positive integer >0
        # self.strides_dim = config['model']['strides_dim'] #: !!int 1 # assumes strides to be same across the two dimensions 
        # self.kernel_initializer = config['model']['kernel_initializer'] 
        # self.recurrent_initializer = config['model']['recurrent_initializer']
        # self.padding = config['model']['padding']
        # self.conv_activation = config['model']['conv_activation']
        # self.recurrent_activation = config['model']['recurrent_activation']
        self.num_heads = config['model']['num_heads']
        self.key_dim = config['model']['key_dim']

    def compile(self):
        tf.random.set_seed(self.seed)

        time_steps = self.window_size
        _, window, height, width, _ = self.x_iv_train.shape
        channels = 1

        inputs = tf.keras.Input(shape=(time_steps, height, width, channels))
        x = tf.keras.layers.Reshape((time_steps, height * width))(inputs)  # flatten spatial grid
        x = tf.keras.layers.Dense(64)(x)  # project to embedding dim

        def create_causal_mask(seq_len):
            return np.triu(np.ones((seq_len, seq_len)), k=1)

        # When calling the self-attention layer
        mask = create_causal_mask(time_steps)  # time_steps is the sequence length

        self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            name="self_attention",
        )

        attention_output, attention_scores = self_attention(
            x, x, attention_mask=mask, return_attention_scores=True
        )


        x = tf.keras.layers.Add()([x, attention_output])
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(height * width)(x)
        outputs = tf.keras.layers.Reshape((height, width, 1))(x[:, -1])  # predict for last time step

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.model.compile(loss=masked_mse, optimizer=self.optimizer)

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
    

    def get_attention(self):
        sample_input = self.x_iv_val[0:1]  #(1, time, height, width, channels)

        flattened = tf.reshape(sample_input, (1, sample_input.shape[1], -1))  # (1, time, H*W)

        dense = tf.keras.layers.Dense(64)  
        embedded = dense(flattened)

        # attention layer
        attention_layer = self.model.get_layer("self_attention")
        attention_output, attn_weights = attention_layer(
            embedded, embedded, return_attention_scores=True)
        
        return attn_weights
    
    def pred(self, x_iv): 
        pred = self.model.predict(x_iv)
        return pred
    