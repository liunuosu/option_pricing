import pandas as pd 
from utils import get_config, print_config
from utils.dataloader import load_data, frame_to_numpy
import yaml
import time
from datetime import datetime
from model import convLSTM


def main(config_file):

    # print all information before starting the run
    print("Configuration Parameters:")
    print_config(config_file)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print('start:', timestamp)
    start_time = time.time()

    # RUN MODEL HERE, input as x_train, y_train, and x_val, y_val!! (x_test, y_test)
    # Must also have same input for the benchmarks

    print("Done training")
    #Retrieve the results, and write it out to the results folder
    # we must get: total R-squared and IVRMSE for validation set
    # Total R-squared and IVRMSE for test set, and time R-sq and IVRMSE
    
    #After that, think about the extensions. Covariate and self-attention!
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nTime taken to run the code: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    config = get_config()
    main(config)

    
