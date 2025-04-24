import tensorflow as tf

def calculate_ivrmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def calculate_r_oos(y_true, y_pred, all_points=False):
    if not all_points:
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
        mean_IV = tf.reduce_mean(y_true, axis=[1, 2], keepdims=True) # should be shape of 114 long
        # print(mean_IV[0], mean_IV[1], mean_IV[2])
        # print(y_true[0], y_true[1], y_true[2])
        ss_tot = tf.reduce_sum(tf.square(y_true - mean_IV))
        # print((y_true - mean_IV)[0])
        r2 = 1 - ss_res/ss_tot
    else:
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred), axis=[1, 2])
        mean_IV = tf.reduce_mean(y_true, axis=[1, 2], keepdims=True)
        ss_tot = tf.reduce_sum(tf.square(y_true - mean_IV), axis=[1, 2])
        r2 = 1 - ss_res/ss_tot
    return r2.numpy()

def calculate_ivrmse_mask(y_true, y_pred, all_points=False):
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.double)
    squared_diff = tf.square(y_true - y_pred) * mask
    
    if not all_points:
        mean = tf.reduce_sum(squared_diff) / tf.reduce_sum(mask)
        ivrmse = tf.sqrt(mean)
    else:
        mean = tf.reduce_sum(squared_diff, axis=[1,2]) / tf.reduce_sum(mask, axis=[1,2])
        ivrmse = tf.sqrt(mean)

    return ivrmse.numpy()

def calculate_r_oos_mask_train(y_true, y_pred, y_train, all_points=False):
    mask = tf.cast(y_true > 0, tf.double)
    train_mask = tf.cast(y_train > 0, tf.double)

    numerator = tf.reduce_sum(y_train * train_mask)  # shape: [H, W]
    denominator = tf.reduce_sum(train_mask)          # shape: [H, W]
    mean_IV = numerator / denominator  # shape: [H, W]

    if not all_points:
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred) * mask)
        ss_tot = tf.reduce_sum(tf.square(y_true - mean_IV) * mask)
        r2 = 1 - ss_res / ss_tot
    else:
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred) * mask, axis=[1, 2])
        ss_tot = tf.reduce_sum(tf.square(y_true - mean_IV) * mask, axis=[1, 2])
        r2 = 1 - ss_res / ss_tot

    return r2.numpy()


def calculate_r_oos_mask_test(y_true, y_pred, y_train, all_points=False):
    mask = tf.cast(y_true > 0, tf.double)
    
    numerator = tf.reduce_sum(y_true * mask)  
    denominator = tf.reduce_sum(mask)  # BUG HERE, was y_true, should be # mask!!!!!    
    mean_IV = numerator / denominator  

    if not all_points:
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred) * mask)
        ss_tot = tf.reduce_sum(tf.square(y_true - mean_IV) * mask)
        r2 = 1 - ss_res / ss_tot
    else:
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred) * mask, axis=[1, 2])
        ss_tot = tf.reduce_sum(tf.square(y_true - mean_IV) * mask, axis=[1, 2])
        r2 = 1 - ss_res / ss_tot

    return r2.numpy()


def calculate_r_oos_mask(y_true, y_pred, all_points=False):

    mask = tf.cast(y_true > 0, tf.double)

    if not all_points:
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred) * mask)

        mean_IV = tf.reduce_sum(y_true * mask, axis = [1, 2], keepdims=True) / tf.reduce_sum(mask, axis=[1, 2], keepdims=True)
        ss_tot = tf.reduce_sum(tf.square(y_true - mean_IV) * mask)
        r2 = 1 - ss_res/ss_tot
    else:
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred)* mask, axis=[1, 2])
        mean_IV = tf.reduce_sum(y_true * mask, axis = [1, 2], keepdims=True) / tf.reduce_sum(mask, axis=[1, 2], keepdims=True)
        ss_tot = tf.reduce_sum(tf.square(y_true - mean_IV) * mask, axis=[1, 2])
        r2 = 1 - ss_res/ss_tot
    return r2.numpy()

