import json
import pickle
import sys
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def model_evaluation() -> None:
    """
    Evaluates the trained model on the test set and saves the metrics.
    """
    try:
        logging.info("Loading model from model.pkl")
        with open('./models/model.pkl', 'rb') as f:
            clf = pickle.load(f)
        
        logging.info("Loading test data")
        test_data = pd.read_csv('./data/processed/test_bow.csv')

        X_test = test_data.iloc[:, 0:-1].values
        y_test = test_data.iloc[:, -1].values

        logging.info("Making predictions on the test set")
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        logging.info("Calculating evaluation metrics")
        accuracy: float = accuracy_score(y_test, y_pred)
        precision: float = precision_score(y_test, y_pred)
        recall: float = recall_score(y_test, y_pred)
        auc: float = roc_auc_score(y_test, y_pred_proba)
        
        metrics_dict: Dict[str, float] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        
        logging.info(f"Metrics: {metrics_dict}")
        
        logging.info("Saving metrics to metrics.json")
        with open('./reports/metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        
        logging.info("Model evaluation completed successfully")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}. Please run the previous pipeline steps.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred during model evaluation: {e}")
        sys.exit(1)

if __name__ == '__main__':
    model_evaluation()