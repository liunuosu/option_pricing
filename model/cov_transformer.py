import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import ConvLSTM2D, LSTM, BatchNormalization, Conv3D, Conv2D,Input, Dense, Reshape, Concatenate
from keras.layers import Masking, Embedding
from keras.layers import LayerNormalization, MultiHeadAttention, Dropout, Add
from utils.loss import masked_mse
import numpy as np

class CovTransformer:

    def __init__(self, x_iv_train, x_cov_train, y_iv_train, \
                 x_iv_val=None, x_cov_val=None, y_iv_val=None, config=None):
        self.read_config(config) 
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

        self.num_heads = config['model']['num_heads']
        self.key_dim = config['model']['key_dim']

    def compile(self):
        tf.random.set_seed(self.seed)

        time_steps = self.window_size
        _, window, height, width, _ = self.x_iv_train.shape
        num_covariates = len(self.covariate_columns)
        patch_dim = height * width

        iv_input = Input(shape=(time_steps, height, width, 1), name="iv_input")
        cov_input = Input(shape=(time_steps, num_covariates), name="cov_input")
        
        x_iv = tf.reshape(iv_input, [-1, time_steps, patch_dim])  # (B, T, H*W)
        x_iv = Dense(64)(x_iv)  # feature space
        x_iv = LayerNormalization()(x_iv)

        positions = tf.range(start=0, limit=time_steps, delta=1)
        pos_encoding = Embedding(input_dim=time_steps, output_dim=64)(positions) 
        x_iv += pos_encoding  

        def create_causal_mask(seq_len):
           return np.triu(np.ones((seq_len, seq_len)), k=1)

        # # When calling the self-attention layer
        mask = create_causal_mask(time_steps)  # time_steps is the sequence length

        # self_attention = tf.keras.layers.MultiHeadAttention(
        #     num_heads=self.num_heads,
        #     key_dim=self.key_dim,
        #     name="self_attention",
        # )

        # attention_output, attention_scores = self_attention(
        #     x, x, attention_mask=mask, return_attention_scores=True
        # )

        #Transformer 
        attn_output = MultiHeadAttention(num_heads=self.num_heads, 
                                         key_dim=self.key_dim)(x_iv, x_iv, attention_mask = mask) 
        x = Add()([x_iv, attn_output])
        x = LayerNormalization()(x)
        #Feedforward layer
        ff = Dense(128, activation='relu')(x)
        ff = Dense(64)(ff)
        x = Add()([x, ff])
        x = LayerNormalization()(x)

        x_iv_out = x[:, -1, :]  
        x_cov = LSTM(units=64, return_sequences=False)(cov_input)
        x_cov = Dense(units=64, activation='relu')(x_cov)

        # concat outputs
        x = Concatenate()([x_iv_out, x_cov])
        x = Dense(units=patch_dim, activation='relu')(x)
        x = Reshape((height, width, 1))(x)

        x = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same')(x)

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
    