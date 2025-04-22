import pandas as pd 
from utils import get_config, print_config, get_results, write_results
from utils.dataloader import dataloader, load_data
from utils.loss import plot_loss
import yaml
import time
from datetime import datetime
from model.ahbs import AHBS
from pathlib import Path
import os
import numpy as np

# Load in the data, no need for validation data

# start with, put or call

# puts
data_train = pd.read_csv("data/final/smoothed/data_train_val.csv")
data_test = pd.read_csv("data/final/evaluation/test_set.csv")