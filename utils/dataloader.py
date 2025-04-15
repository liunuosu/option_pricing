import pandas as pd
import numpy as np

def load_data(run, covariate_columns):
        
    if run == 'short_ttm':
        data_train = pd.read_csv('data/final/smoothed/data_train.csv')
        data_val = pd.read_csv('data/final/evaluation/validation_set.csv')
        data_test = pd.read_csv('data/final/evaluation/test_set.csv')

        if covariate_columns is not None:
            covar_df = pd.read_excel('data/final/covariates/covariates_train.xlsx')
            covar_df_val = pd.read_excel('data/final/covariates/covariates_validation.xlsx')

            covar_df = covar_df.rename(columns={'Date':'date'})
            covar_df_val = covar_df_val.rename(columns={'Date':'date'})
            covar_df = covar_df[['date'] + covariate_columns]
            covar_df_val = covar_df_val[['date'] + covariate_columns]

    elif run == 'long_ttm':
        data_train = pd.read_csv('data/final/smoothed/data_train_long.csv')
        data_val = pd.read_csv('data/final/evaluation/validation_set_long.csv')
        data_test = pd.read_csv('data/final/evaluation/test_set_long.csv')

        if covariate_columns is not None:
            covar_df = pd.read_excel('data/final/covariates/covariates_train_long.xlsx')
            covar_df_val = pd.read_excel('data/final/covariates/covariates_validation_long.xlsx')

            covar_df = covar_df.rename(columns={'Date':'date'})
            covar_df_val = covar_df_val.rename(columns={'Date':'date'})
            covar_df = covar_df[['date'] + covariate_columns]
            covar_df_val = covar_df_val[['date'] + covariate_columns]
    
    else:
        print('Select a dataset')

    return data_train, data_val, data_test

def frame_to_numpy(data, covariate_cols=None, eval=False):
    
    data['time_step'] = data['date']
    time_step_index = pd.to_datetime(data['time_step']).dt.strftime('%Y-%m-%d').unique()
    date_to_index = {date: idx for idx, date in enumerate(time_step_index)}

    data['time_step_str'] = data['time_step'].dt.strftime('%Y-%m-%d')
    data['time_step_idx'] = data['time_step_str'].map(date_to_index)
    #data['time_step_idx'] = data['time_step'].apply(lambda x: np.where(time_step_index == x.strftime('%Y-%m-%d'))[0][0])

    maturity_values = np.sort(data['maturity'].unique())
    maturity_to_idx = {mat: i for i, mat in enumerate(maturity_values)}

    time_steps = len(time_step_index)
    money_dim = len(data['moneyness_enc'].unique())
    ttm_dim = len(maturity_values)

    # Base IV tensor
    IV_array = np.zeros((time_steps, money_dim, ttm_dim, 1))
    cov_array = np.zeros((time_steps, len(covariate_cols)))

    for idx, row in data.iterrows():
        time_step_idx = row['time_step_idx']
        height = int(row['moneyness_enc']) - 1 
        width = maturity_to_idx[row['maturity']]
        value = row['IV_smooth'] if not eval else row['impl_volatility']
        IV_array[time_step_idx, height, width, 0] = value

        for i, cov in enumerate(covariate_cols):
            cov_array[time_step_idx,i] = row[cov]

    return IV_array, cov_array

def create_rolling_window_dataset(iv_array, cov_array, window_size):

    T = iv_array.shape[0]
    N = T - window_size

    x_iv = np.zeros((N, window_size, *iv_array.shape[1:]))        # (N, window_size, H, W, 1)
    x_cov = np.zeros((N, window_size, cov_array.shape[1]))        # (N, window_size, C)
    y = np.zeros((N, *iv_array.shape[1:]))                   # (N, H, W, 1)

    for i in range(N):
        x_iv[i] = iv_array[i:i+window_size]
        x_cov[i] = cov_array[i:i+window_size]
        y[i] = iv_array[i+window_size]

    return x_iv, x_cov, y



x_iv_train, x_cov_train, target_train = create_rolling_window_dataset(IV_train, cov_train, window_size)

IV_val_input = np.concatenate((IV_train[-window_size:], IV_val), axis=0)
cov_val_input = np.concatenate((cov_train[-window_size:], cov_val), axis=0)
x_iv_val, x_cov_val, target_val = create_rolling_window_dataset(IV_val_input, cov_val_input, window_size)

IV_test_input = np.concatenate((IV_val[-window_size:], IV_test), axis=0)
cov_test_input = np.concatenate((cov_val[-window_size:], cov_test))
x_iv_test, x_cov_test, target_test = create_rolling_window_dataset(IV_test_input, cov_test_input, window_size)