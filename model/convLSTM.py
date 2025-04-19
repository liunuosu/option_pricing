from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from pathlib import Path
from utils.loss import masked_mse

def create_model(n_params, 
                 dropout, 
                 recurrent_dropout, 
                 n_convlstm_layers = 2,
                 hidden_activation =  tf.keras.activations.tanh, 
                 optimizer = keras.optimizers.Adam()):

    # input layer
    input_layer = layers.Input(shape= (None,len(expiries),len(params),1) )
    
    # lstm layers
    lstm = input_layer
    for i in range( n_convlstm_layers ):
        lstm =  layers.ConvLSTM2D( 
            kernel_size= (1,1), 
            filters=n_params, 
            data_format= 'channels_last', 
            return_sequences = i<n_convlstm_layers-1,
            activation=hidden_activation,
            padding = "same",
            dropout=dropout, 
            recurrent_dropout=recurrent_dropout
        )( lstm )
        lstm = layers.BatchNormalization()(lstm)    

    output = layers.Conv2D(
        filters=1, kernel_size=(1, 1), activation="linear", padding="same"
    )( lstm )
    output_layer = layers.Reshape( (len(expiries),len(params)) )(output)

    # compile
    model = Model( input_layer, output_layer )
    model.compile(
        loss= "MAE",
        optimizer=optimizer, 
    ) 
    
    print(model.summary())
    return model

def train_model(model, 
                dataset, 
                verbose = True, 
                save : "dir" = False,
                training_kwarg_overwrites : "dict" = {} ):
    
    # train until we run out of improvement
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15),
    ]
    
    # train model
    training_kwargs = {
        "x" : dataset["train"]["x_scaled"],
        "y" : dataset["train"]["y_scaled"],
        "epochs" : 200,
        "batch_size" : 64,
        "verbose" : verbose,
        "validation_split" : 0.2,
        "callbacks" : callbacks,
    } 
    training_kwargs.update(training_kwarg_overwrites)
    train_hist = model.fit( **training_kwargs )
    
    if save:
        Path(save).mkdir(parents=True, exist_ok=True) # make a home for the models
        train_start, train_end = [ f( dataset["dates"]["train"] ) for f in (min,max) ]
        model_name = "-".join( date.strftime("%Y%m%d") for date in [train_start, train_end] )
        model.save( save+model_name )
        
    return model, train_hist

