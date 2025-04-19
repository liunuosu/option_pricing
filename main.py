import pandas as pd 
from utils import get_config, print_config, get_results, write_results
from utils.dataloader import dataloader
from utils.loss import plot_loss
import yaml
import time
from datetime import datetime
from model.cov_convlstm import CovConvLSTM
from pathlib import Path

def main(config_file):
    # print all information before starting the run
    print("Configuration Parameters:")
    print_config(config_file)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print('start:', timestamp)
    start_time = time.time()
    print("Loading data..")
    
    data_config = config['data']
    run = data_config['run']
    option_type = data_config["option"]
    smooth = data_config["smooth"]
    full_train = data_config["full_train"]
    covariate_columns = data_config["covariates"]
    window_size = data_config['window_size']
    h_step = data_config['h']

    temporary_map = 'data/final/binned'

    # Should return the train, validation and test data for the covariates 
    
    if not full_train:
        x_iv_train, x_cov_train, target_train, x_iv_val, x_cov_val, \
            target_val, x_iv_test, x_cov_test, target_test, IV_val, IV_test = \
                dataloader(run, option_type, smooth, full_train, covariate_columns, window_size, h_step, folder_path=temporary_map)
    else:
        x_iv_train, x_cov_train, target_train, x_iv_test, x_cov_test, target_test, IV_test = \
            dataloader(run, option_type, smooth, full_train, covariate_columns, window_size, h_step, folder_path=temporary_map)
    print("Done loading data")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to load data: {elapsed_time:.2f} seconds")

    # First just make the code work for covariates too
    print("Initializing and training model")
    conv_lstm = CovConvLSTM(x_iv_train, x_cov_train, target_train, x_iv_val, x_cov_val, target_val, config_file)
    conv_lstm.compile()
    best_epoch, best_loss, train_loss, val_loss = conv_lstm.fit()

    end2 = time.time()
    elapsed_time = end2- end_time
    print(f"Time taken to train model: {elapsed_time:.2f} seconds")
    print(f"Best epoch: {best_epoch}, best loss: {best_loss}")

    if not full_train:
        pred_val = conv_lstm.pred(x_iv_val, x_cov_val)
        folder_path = Path(f"results/validation_{run}")
        ivrmse, ivrmse_h, r_oos, r_oos_h = get_results(IV_val[h_step-1:], pred_val)
        write_results(folder_path, ivrmse, r_oos, ivrmse_h, r_oos_h, IV_val[h_step-1:], pred_val, covariate_columns, option_type)

    pred_test = conv_lstm.pred(x_iv_test, x_cov_test)
    folder_path = Path(f"results/test_{run}")
    ivrmse, ivrmse_h, r_oos, r_oos_h = get_results(IV_test[h_step-1:], pred_test)
    write_results(folder_path, ivrmse, r_oos, ivrmse_h, r_oos_h, IV_test[h_step-1:], pred_test, covariate_columns, option_type)

    plot_loss(train_loss, val_loss)
if __name__ == "__main__":
    config_name = 'config_file_covs.yaml'
    config = get_config()
    main(config)

    
