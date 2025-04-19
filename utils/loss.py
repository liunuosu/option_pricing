import tensorflow as tf
import matplotlib.pyplot as plt

def masked_mse(y_true, y_pred):
    # obtain true and false objects using tf.not_equal. If it is true, then cast them to 1s, otherwise it is 0
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    # print(mask)

    #Multiply square differences with the mask, so only the 1s are taken into account
    squared_diff = tf.square(y_true - y_pred) * mask

    # sum the amount of 1s
    safe_mask_sum = tf.reduce_sum(mask)

    # sum the amount of square diffences, and divide by total amount 
    return tf.reduce_sum(squared_diff) / safe_mask_sum

def plot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='Training Loss')
    if val_loss:
        plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()