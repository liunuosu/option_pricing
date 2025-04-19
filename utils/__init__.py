import yaml
from pathlib import Path
from metrics import calculate_ivrmse_mask, calculate_r_oos_mask
import numpy as np

def get_config(config_name):
    with(open(Path("configs", config_name))) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config

def print_config(config):
    for branch_name, branch_params in config.items():
        print(branch_name +':', config[branch_name])
        
def get_results(y_real, y_pred):
    ivrmse = calculate_ivrmse_mask(y_real, y_pred)
    ivrmse_h = calculate_ivrmse_mask(y_real, y_pred, all_points=True)
    r_oos = calculate_r_oos_mask(y_real, y_pred)
    r_oos_h = calculate_r_oos_mask(y_real, y_pred, all_points=True)

    return ivrmse, ivrmse_h, r_oos, r_oos_h

def write_results(folder_path, ivrmse, r_oos, ivrmse_h, r_oos_h, surface, surface_pred, covariate_columns, option_type, smooth, window_size, h_step):

    ivrmse_path = folder_path / Path("ivrmse")
    r_oos_path = folder_path / Path("r_oos")
    ivrmse_h_path = folder_path / Path("ivrmse_h")
    r_oos_h_path = folder_path / Path("r_oos_h")
    surface_path = folder_path / Path("surface")
    surface_pred_path = folder_path / Path("surface_pred")

    if not ivrmse_path.exists():
        ivrmse_path.mkdir(parents=True, exist_ok=True)

    if not r_oos_path.exists():
        r_oos_path.mkdir(parents=True, exist_ok=True)

    if not ivrmse_h_path.exists():
        ivrmse_h_path.mkdir(parents=True, exist_ok=True)

    if not r_oos_h_path.exists():
        r_oos_h_path.mkdir(parents=True, exist_ok=True)

    if not surface_path.exists():
        surface_path.mkdir(parents=True, exist_ok=True)

    if not surface_pred_path.exists():
        surface_pred_path.mkdir(parents=True, exist_ok=True)

    cov = ""
    for i in covariate_columns:
        cov = cov+ "_" +i
    np.save(ivrmse_path / f"{option_type}_smooth_{smooth}_ws_{window_size}_h_{h_step}{cov}.npy", ivrmse)
    np.save(r_oos_path / f"{option_type}_smooth_{smooth}_ws_{window_size}_h_{h_step}{cov}.npy", r_oos)
    np.save(ivrmse_h_path / f"{option_type}_smooth_{smooth}_ws_{window_size}_h_{h_step}{cov}.npy", ivrmse_h)
    np.save(r_oos_h_path / f"{option_type}_smooth_{smooth}_ws_{window_size}_h_{h_step}{cov}.npy", r_oos_h)
    np.save(surface_path/ f"{option_type}_smooth_{smooth}_ws_{window_size}_h_{h_step}{cov}.npy", surface)
    np.save(surface_pred_path / f"{option_type}_smooth_{smooth}_ws_{window_size}_h_{h_step}{cov}.npy", surface_pred)