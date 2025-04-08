import yaml
from pathlib import Path
from metrics import calculate_ivrmse, calculate_r_oos
import numpy as np

def get_config():
    with(open(Path("configs", "config_file.yaml"))) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config

def print_config(config):
    for branch_name, branch_params in config.items():
        print(branch_name +':', config[branch_name])
        
def get_results(y_real, y_pred):
    ivrmse = calculate_ivrmse(y_real, y_pred)
    ivrmse_h = calculate_ivrmse(y_real, y_pred, all_points=True)
    r_oos = calculate_r_oos(y_real, y_pred)
    r_oos_h = calculate_r_oos(y_real, y_pred)

    return ivrmse, ivrmse_h, r_oos, r_oos_h

def write_results(folder_path, ivrmse, r_oos, ivrmse_h, r_oos_h, surface, surface_pred, window_size, h_step):

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

    np.save(ivrmse_path / f"{window_size}_{h_step}.npy", ivrmse)
    np.save(r_oos_path / f"{window_size}_{h_step}.npy", r_oos)
    np.save(ivrmse_h_path / f"{window_size}_{h_step}.npy", ivrmse_h)
    np.save(r_oos_h_path / f"{window_size}_{h_step}.npy", r_oos_h)
    np.save(surface_path/ f"{window_size}_{h_step}.npy", surface)
    np.save(surface_pred_path / f"{window_size}_{h_step}.npy", surface_pred)
