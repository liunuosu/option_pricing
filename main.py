import pandas as pd 
from utils import get_config, print_config
import yaml
import time
from datetime import datetime



def main(config_file):

    # print all information before starting the run
    print("Configuration Parameters:")
    print_config(config_file)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print('start:', timestamp)
    start_time = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nTime taken to run the code: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    config = get_config()
    main(config)

    
