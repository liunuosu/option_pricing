import pandas as pd 
from utils import get_config, print_config, get_results, write_results
from utils.dataloader import dataloader
from utils.loss import plot_loss
import yaml
import time
from datetime import datetime
from model.cov_convlstm import CovConvLSTM
from model.convlstm import ConvLSTM
from model.transformer import Transformer
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt

def main(config_file):
    # print all information before starting the run
    print("Configuration Parameters:")
    print_config(config_file)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print('start:', timestamp)
    start_time = time.time()
    print("Loading data..")
    
    data_config = config_file['data']
    run = data_config['run']
    option_type = data_config["option"]
    smooth = data_config["smooth"]
    full_train = data_config["full_train"]
    covariate_columns = data_config["covariates"]
    window_size = data_config['window_size']
    h_step = data_config['h_step']

    temporary_map = 'data/final/binned'

    # Should return the train, validation and test data for the covariates 
    
    if not full_train:
        x_iv_train, x_cov_train, target_train, x_iv_val, x_cov_val, \
            target_val, x_iv_test, x_cov_test, target_test, IV_train, IV_val, IV_test = \
                dataloader(run, option_type, smooth, full_train, covariate_columns, window_size, 
                           h_step, folder_path=temporary_map)
    else:
        x_iv_train, x_cov_train, target_train, x_iv_test, x_cov_test, target_test, IV_train, IV_test = \
            dataloader(run, option_type, smooth, full_train, covariate_columns, window_size, h_step, 
                       folder_path=temporary_map)
    print("Done loading data")
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken to load data: {int(minutes)} minutes {seconds:.2f} seconds")

    # First just make the code work for covariates too
    print("Initializing and training model")
    note = config_file['model']['note']

    transformer = Transformer(x_iv_train, target_train, x_iv_val, target_val, config_file)
    transformer.compile()
    best_epoch, best_loss, train_loss, val_loss = transformer.fit()

    end2 = time.time()
    elapsed_time = end2- end_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken to train model: {int(minutes)} minutes {seconds:.2f} seconds")
    print(f"Best Epoch: {best_epoch}, best loss: {best_loss:.6f}")

    
    pred_val = transformer.pred(x_iv_val)
    pred_test = transformer.pred(x_iv_test)
    
    folder_path = Path(f"results/validation_{run}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    ivrmse, ivrmse_h, r_oos, r_oos_h = get_results(IV_val[h_step-1:], pred_val, IV_train)
    write_results(folder_path, ivrmse, r_oos, ivrmse_h, r_oos_h, IV_val[h_step-1:], 
                    pred_val, covariate_columns, option_type, smooth, window_size, h_step, note)

    folder_path = Path(f"results/test_{run}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    ivrmse, ivrmse_h, r_oos, r_oos_h = get_results(IV_test[h_step-1:], pred_test, IV_train)
    # print(ivrmse, r_oos)
    write_results(folder_path, ivrmse, r_oos, ivrmse_h, r_oos_h, IV_test[h_step-1:], 
                      pred_test, covariate_columns, option_type, smooth, window_size, h_step, note)

    weights = transformer.get_attention()
    np.save(f"results/attention_weights/{run}_{option_type}_sm_{smooth}_ws_{window_size}_h_{h_step}_{note}", weights)
    # print(weights)

    # np.save("weights", weights.numpy())

    # plt.imshow(weights[0,0].numpy(), cmap='coolwarm')
    # plt.title("Attention Head 0")
    # plt.xlabel("Key timestep")
    # plt.ylabel("Query timestep")
    # plt.colorbar()
    # plt.show()

if __name__ == "__main__":
    config_original = 'config_file_transformer.yaml'
    config = get_config(config_original)
    for i in ['short_ttm']:
        for j in ['call', 'put']:
            for l in [5, 21, 63]:
                for k in [1, 5, 10]:
                    for m in [True]:
                        for n in [4, 8, 16]:
                            for o in [16, 32, 64]:
                                config['data']['run'] = i
                                config['data']['option'] = j
                                config['data']['window_size'] = l
                                config['data']['h_step'] = k
                                config['data']['smooth'] = m
                                config['model']['num_heads'] = n
                                config['model']['key_dim'] = o
                                config['model']['note'] = f"{n}_{o}_transformerNEWv2"
                                main(config)


    
