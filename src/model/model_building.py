import pickle
import sys
import logging
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from typing import Text, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def model_building() -> None:
    """
    Trains the model using the feature set and saves the trained model.
    """
    try:
        logging.info("Loading parameters from params.yaml")
        with open('params.yaml', 'r') as f:
            params: Dict[str, Any] = yaml.safe_load(f)['model_building']
        
        logging.info("Loading training data")
        train_data = pd.read_csv('./data/processed/train_bow.csv')

        X_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values

        logging.info("Training Gradient Boosting Classifier")
        clf = GradientBoostingClassifier(
            n_estimators=params['n_estimator'],
            learning_rate=params['learning_rate']
        )
        clf.fit(X_train, y_train)
        
        logging.info("Saving model to model.pkl")
        with open('./models/model.pkl', 'wb') as f:
            pickle.dump(clf, f)

        logging.info("Model training and saving completed successfully")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}. Please run the feature engineering step.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred during model building: {e}")
        sys.exit(1)

if __name__ == '__main__':
    model_building()