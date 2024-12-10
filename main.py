import pandas as pd 
from utils import get_config

def main(config_file):
    print('wetji')

if __name__ == "__main__":
    config = get_config()
    print(config['training'])

    main(config)

    
