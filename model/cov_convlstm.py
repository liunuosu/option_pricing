import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential, Model
from keras.layers import ConvLSTM2D, LSTM, BatchNormalization, Conv3D, Conv2D,Input, Dense, Reshape, Concatenate
from keras.layers import Masking

class CovConvLSTM:

    self.model

    def __init__(time_steps, height, width, num_covariates, learning_rate, epsilon):
        iv_input = Input(shape=(time_steps, height, width, 1), name="iv_input")
        x_iv = Masking(mask_value=0.0)(iv_input)
        x_iv = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(iv_input)
        x_iv = BatchNormalization()(x_iv)
        x_iv = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False)(x_iv)
        x_iv = BatchNormalization()(x_iv)

        cov_input = Input(shape=(time_steps, num_covariates), name="cov_input")

        x_cov = LSTM(units=64, return_sequences=False)(cov_input)   # (batch_size, 64)
        x_cov = Dense(units=height * width, activation='relu')(x_cov)  
        x_cov = Reshape((height, width, 1))(x_cov)               # (batch_size, H, W, 1)

        x = Concatenate(axis=-1)([x_iv, x_cov])  # Combine along channel axis -> (H, W, 65)
        x = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same')(x)

        self.model = Model(inputs=[iv_input, cov_input], outputs=x)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
        self.model.compile(loss=masked_mse, optimizer=optimizer)
        print(self.model.summary())

    def fit(x_iv_train, x_cov_train, target_train, x_iv_val, x_cov_val, target_val, epochs, batch_size, patience):
        history = self.model.fit([x_iv_train, x_cov_train], target_train,
                validation_data=([x_iv_val, x_cov_val], target_val),
                epochs=epochs, batch_size=batch_size, shuffle=False,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            mode='min')])

    def pred(x_iv, x_cov):
        return 0