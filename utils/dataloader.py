import pandas as pd
import numpy as np
import os
from pathlib import Path

def retrieve_data(run, filename, folder_path, raw, covar_df, smooth=False):

    # check if specific file exists. If so, just load them. if not, then compute the whole thing
    if os.path.isfile(filename):
        data = pd.read_csv(filename)
    else:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # Make train data
        df = drop_settlement_dup(raw)

        if run =='long_ttm':
            bins = [0, 21, 63, 126, 189, 252]
            labels = [1, 2, 3, 4, 5] # bin the maturities if it is a long term ttm
            df['maturity'] = pd.cut(df['maturity'], bins=bins, labels=labels, include_lowest=True, right=True).astype('Int64') 
            df = df.dropna(subset=['maturity'])
        
        moneyness_grid = np.arange(0.80, 1.21, 0.05)
        data = bin_avg(df, moneyness_grid, train=smooth)
        data.to_csv(filename)
    
    data['date'] = pd.to_datetime(data['date'])
    if covar_df is not None:
        data = pd.merge(data, covar_df, on='date', how='left')
    data = data.dropna()

    return data

# Should return the train, validation and test data for the covariates 
def dataloader(run, option_type, smooth, full_train, covariate_columns, window_size, h_step, folder_path='data/final/binned'):

    if not full_train: 
        train_name = f"data/final/binned/train_{run}_{option_type}_{smooth}.csv"
        val_name = f"data/final/binned/val_{run}_{option_type}.csv"
        test_name = f"data/final/binned/test_{run}_{option_type}.csv"

        train_raw, val_raw, test_raw, covar_df = load_data(run, option_type, covariate_columns)

        # Use the smooth parameter, depends on the config file if it is used or not
        # only train data is smoothed to avoid data leakage
        data_train = retrieve_data(run, train_name, folder_path, train_raw, covar_df, smooth)
        data_val = retrieve_data(run, val_name, folder_path, val_raw, covar_df)
        data_test = retrieve_data(run, test_name, folder_path, test_raw, covar_df)

        IV_train, cov_train = frame_to_numpy(data_train, covariate_columns)
        IV_val, cov_val = frame_to_numpy(data_val, covariate_columns)
        IV_test, cov_test = frame_to_numpy(data_test, covariate_columns)

        x_iv_train, x_cov_train, target_train = create_rolling_window_dataset(IV_train, cov_train, window_size, h_step)

        IV_val_input = np.concatenate((IV_train[-window_size:], IV_val), axis=0)
        cov_val_input = np.concatenate((cov_train[-window_size:], cov_val), axis=0)
        x_iv_val, x_cov_val, target_val = create_rolling_window_dataset(IV_val_input, cov_val_input, window_size, h_step)

        IV_test_input = np.concatenate((IV_val[-window_size:], IV_test), axis=0)
        cov_test_input = np.concatenate((cov_val[-window_size:], cov_test))
        x_iv_test, x_cov_test, target_test = create_rolling_window_dataset(IV_test_input, cov_test_input, window_size, h_step)


        return x_iv_train, x_cov_train, target_train, x_iv_val, x_cov_val, \
            target_val, x_iv_test, x_cov_test, target_test, IV_val, IV_test
    else:   
        train_name = f"data/final/binned/train_full_{run}_{option_type}_{smooth}.csv"
        test_name = f"data/final/binned/test_{run}_{option_type}.csv"
        train_raw, test_raw, covar_df_val = load_data(run, option_type, covariate_columns, full_train)
        
        data_train = retrieve_data(run, train_name, folder_path, train_raw, covar_df_val, smooth)
        data_test = retrieve_data(run, test_name, folder_path, test_raw, covar_df_val)

        IV_train, cov_train = frame_to_numpy(data_train, covariate_columns)
        IV_test, cov_test = frame_to_numpy(data_test, covariate_columns)

        x_iv_train, x_cov_train, target_train = create_rolling_window_dataset(IV_train, cov_train, window_size, h_step)

        IV_test_input = np.concatenate((IV_train[-window_size:], IV_test), axis=0)
        cov_test_input = np.concatenate((cov_train[-window_size:], cov_test))
        x_iv_test, x_cov_test, target_test = create_rolling_window_dataset(IV_test_input, cov_test_input, window_size, h_step)

        return x_iv_train, x_cov_train, target_train, x_iv_test, \
            x_cov_test, target_test, IV_test

def load_data(run, option_type, covariate_columns, full_train=False):
        
    # Let's reshape our input data of the thing... we are going to need labels, and we are going to need train surface.
    # The labels will be, the smoothed IVs of our data
    # The train will be the dimensions, with time x ttm x moneyness encoders
    # If we have covariates, the channels will be larger? -> Yes, starting channels will be added to the layers
    # Load the data for train/val/test

    if not full_train:
        if run == 'short_ttm':
            data_train = pd.read_csv('data/final/smoothed/data_train.csv')
            data_val = pd.read_csv('data/final/evaluation/validation_set.csv')
            data_test = pd.read_csv('data/final/evaluation/test_set.csv')

            if option_type =='put':
                data_train[data_train['cp_flag']=='P']
                data_val[data_val['cp_flag']=='P']
                data_test[data_test['cp_flag']=='P']
            elif option_type =='call':
                data_train[data_train['cp_flag']=='C']
                data_val[data_val['cp_flag']=='C']
                data_test[data_test['cp_flag']=='C']

            if covariate_columns:
                covar_df = pd.read_excel('data/final/covariates/covariates_train.xlsx')
                covar_df_val = pd.read_excel('data/final/covariates/covariates_validation.xlsx')

                covar_df = covar_df.rename(columns={'Date':'date'})
                covar_df_val = covar_df_val.rename(columns={'Date':'date'})
                covar_df = covar_df[['date'] + covariate_columns]
                covar_df_val = covar_df_val[['date'] + covariate_columns]
            else:
                covar_df = None

        elif run == 'long_ttm':
            data_train = pd.read_csv('data/final/smoothed/data_train_long.csv')
            data_val = pd.read_csv('data/final/evaluation/validation_set_long.csv')
            data_test = pd.read_csv('data/final/evaluation/test_set_long.csv')

            if option_type =='put':
                data_train[data_train['cp_flag']=='P']
                data_val[data_val['cp_flag']=='P']
                data_test[data_test['cp_flag']=='P']
            elif option_type =='call':
                data_train[data_train['cp_flag']=='C']
                data_val[data_val['cp_flag']=='C']
                data_test[data_test['cp_flag']=='C']
                
            if covariate_columns:
                covar_df = pd.read_excel('data/final/covariates/covariates_train_long.xlsx')
                covar_df_val = pd.read_excel('data/final/covariates/covariates_validation_long.xlsx')

                covar_df = covar_df.rename(columns={'Date':'date'})
                covar_df_val = covar_df_val.rename(columns={'Date':'date'})
                covar_df = covar_df[['date'] + covariate_columns]
                covar_df_val = covar_df_val[['date'] + covariate_columns]
            else:
                covar_df = None
        
        else:
            print('Select a dataset')

        return data_train, data_val, data_test, covar_df
    else:
        if run == 'short_ttm':
            data_train = pd.read_csv('data/final/smoothed/data_train_val.csv')
            data_test = pd.read_csv('data/final/evaluation/test_set.csv')

            if option_type =='put':
                data_train[data_train['cp_flag']=='P']
                data_test[data_test['cp_flag']=='P']
            elif option_type =='call':
                data_train[data_train['cp_flag']=='C']
                data_test[data_test['cp_flag']=='C']

            if covariate_columns:
                covar_df_val = pd.read_excel('data/final/covariates/covariates_validation.xlsx')
                covar_df_val = covar_df_val.rename(columns={'Date':'date'})
                covar_df_val = covar_df_val[['date'] + covariate_columns]
            else:
                covar_df_val = None

        elif run == 'long_ttm':
            data_train = pd.read_csv('data/final/smoothed/data_train_val_long.csv')
            data_test = pd.read_csv('data/final/evaluation/test_set_long.csv')

            if option_type =='put':
                data_train[data_train['cp_flag']=='P']
                data_test[data_test['cp_flag']=='P']
            elif option_type =='call':
                data_train[data_train['cp_flag']=='C']
                data_test[data_test['cp_flag']=='C']
                
            if covariate_columns:
                covar_df_val = pd.read_excel('data/final/covariates/covariates_validation_long.xlsx')
                covar_df_val = covar_df_val.rename(columns={'Date':'date'})
                covar_df_val = covar_df_val[['date'] + covariate_columns]
            else:
                covar_df_val = None
        
        else:
            print('Select a dataset')

        return data_train, data_test, covar_df_val

def drop_settlement_dup(df):
    df = df.sort_values(by='am_settlement')
    df = df.drop_duplicates(subset=['maturity', 'moneyness'], keep='first')
    return df

def bin_avg(df, moneyness_grid, bin_width=0.05, train=True):
   
    result = []

    for date in df['date'].unique():
        for maturity in df['maturity'].unique():
            # Filter the data for the current date and maturity
            filtered_df = df[(df['date'] == date) & (df['maturity'] == maturity)]
            
            # Bin each moneyness point within the filtered data
            for center in moneyness_grid:
                lower = center - (bin_width / 2)
                upper = center + (bin_width / 2)

                # Select data points within the current bin
                bin_data = filtered_df[(filtered_df['moneyness'] >= lower) & (filtered_df['moneyness'] < upper)]
                
                if train:
                    avg_iv = bin_data['IV_smooth'].mean() if not bin_data.empty else np.nan
                else:                   
                    avg_iv = bin_data['impl_volatility'].mean() if not bin_data.empty else np.nan

                # Append the result for this combination of date, maturity, and moneyness
                result.append({
                    'date': date,
                    'maturity': maturity,
                    'moneyness': center,
                    'impl_volatility': avg_iv
                })

    # Return the final dataframe with the binned results
    return pd.DataFrame(result)

def frame_to_numpy(data, covariate_cols=None):
    
    data['time_step'] = data['date']
    time_step_index = pd.to_datetime(data['time_step']).dt.strftime('%Y-%m-%d').unique()
    date_to_index = {date: idx for idx, date in enumerate(time_step_index)}

    data['time_step_str'] = data['time_step'].dt.strftime('%Y-%m-%d')
    data['time_step_idx'] = data['time_step_str'].map(date_to_index)
    #data['time_step_idx'] = data['time_step'].apply(lambda x: np.where(time_step_index == x.strftime('%Y-%m-%d'))[0][0])

    maturity_values = np.sort(data['maturity'].unique())
    maturity_to_idx = {mat: i for i, mat in enumerate(maturity_values)}

    time_steps = len(time_step_index)
    money_dim = len(data['moneyness'].unique())
    ttm_dim = len(maturity_values)

    # Base IV tensor
    IV_array = np.zeros((time_steps, money_dim, ttm_dim, 1))
    cov_array = np.zeros((time_steps, len(covariate_cols)))

    for idx, row in data.iterrows():
        time_step_idx = row['time_step_idx']
        height = int(row['moneyness']) - 1 
        width = maturity_to_idx[row['maturity']]
        value = row['impl_volatility']
        IV_array[time_step_idx, height, width, 0] = value

        for i, cov in enumerate(covariate_cols):
            cov_array[time_step_idx,i] = row[cov]

    return IV_array, cov_array


def create_rolling_window_dataset(iv_array, cov_array, window_size, h): # hstep ahead here

    T = iv_array.shape[0]
    N = T - window_size - (h - 1)

    x_iv = np.zeros((N, window_size, *iv_array.shape[1:]))        # (N, window_size, H, W, 1)
    x_cov = np.zeros((N, window_size, cov_array.shape[1]))        # (N, window_size, C)
    y = np.zeros((N, *iv_array.shape[1:]))                   # (N, H, W, 1)

    for i in range(N):
        x_iv[i] = iv_array[i:i+window_size]
        x_cov[i] = cov_array[i:i+window_size]
        y[i] = iv_array[i+window_size + (h - 1)]

    return x_iv, x_cov, y