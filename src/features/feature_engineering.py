import os
import sys
import logging
import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from typing import Text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def feature_engineering() -> None:
    """
    Applies Bag of Words to preprocessed data and saves the feature sets.
    """
    try:
        logging.info("Loading parameters from params.yaml")
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        
        max_features: int = params['feature_engineering']['max_features']
        
        logging.info("Loading preprocessed data")
        # Note: Original file used an absolute path, changed to a relative one for better portability.
        train_data = pd.read_csv('./data/interim/train_processed.csv')
        test_data = pd.read_csv('./data/interim/test_processed.csv')

        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)

        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        
        logging.info(f"Applying Bag of Words with max_features={max_features}")
        vectorizer = CountVectorizer(max_features=max_features)
        
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test
        
        data_path = os.path.join("data", "processed")
        os.makedirs(data_path, exist_ok=True)
        
        train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)

        logging.info("Successfully saved feature data")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}. Please run the previous pipeline steps.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred during feature engineering: {e}")
        sys.exit(1)

if __name__ == '__main__':
    feature_engineering()