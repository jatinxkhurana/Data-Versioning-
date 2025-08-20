import os
import sys
import logging
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from typing import Text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def data_ingestion() -> None:
    """
    Ingests data, splits it, and saves it to the raw data directory.
    """
    try:
        logging.info("Loading parameters from params.yaml")
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        
        test_size: float = params['data_ingestion']['test_size']
        
        logging.info("Fetching data from URL")
        df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

        df.drop(columns=['tweet_id'], inplace=True)

        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        
        logging.info("Splitting data into training and testing sets")
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        data_path = os.path.join('data', 'raw')
        os.makedirs(data_path, exist_ok=True)
        
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        
        logging.info("Successfully saved raw train and test data")

    except FileNotFoundError:
        logging.error("params.yaml not found. Please ensure the file exists.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred during data ingestion: {e}")
        sys.exit(1)

if __name__ == '__main__':
    data_ingestion()